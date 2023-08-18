import math
import torch
import torch.nn as nn
import f16s4_gemm  # with CUDA kernels
from typing import Optional, Tuple


class ScaledActivation(nn.Module):

    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


CLIPMIN = 1e-4


class UniformAffineQuantizer(nn.Module):

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
    ):
        """
        Modified from https://github.com/hahnyuan/RPTQ4LLM
        remove cluster quantize
        remove dynamic_method
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2**n_bits - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None

        self.enable = True
        self.recorded_quant_input = None

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = -(2**(n_bits - 1))
        self.qmax = 2**(n_bits - 1) - 1

    def quant(self,
              x,
              scale,
              round_zero_point,
              uint=False,
              group=-1,
              add_dim=None):
        # Using broadcast for group quantize to avoid too much permute
        org_shape = x.shape
        if group > 0:
            x = x.reshape(x.shape[:-1] + (-1, group))
            scale = scale.reshape(-1, x.shape[-2]).unsqueeze(0).unsqueeze(
                -1)  # Unsqueeze 0 for batch dim, -1 for group dim.
            round_zero_point = round_zero_point.reshape(
                -1, x.shape[-2]).unsqueeze(0).unsqueeze(-1)
            if add_dim:
                scale = scale.unsqueeze(add_dim)
                round_zero_point = round_zero_point.unsqueeze(add_dim)
        x_int = (x / scale).round_()
        if uint:
            x_int = x_int.add_(round_zero_point - self.qmin)
            x_int = x_int.clamp_(0, self.qmax - self.qmin).to(torch.uint8)
            return x_int.reshape(org_shape)
        else:
            if not self.symmetric:
                x_int = x_int.add_(round_zero_point)
            x_int = x_int.clamp_(self.qmin, self.qmax).to(torch.int8)
            return x_int.reshape(org_shape)

    def dequant(self,
                x_int,
                scale,
                round_zero_point,
                uint=False,
                group=-1,
                add_dim=None):
        if not x_int.dtype == scale.dtype:
            x_int = x_int.to(scale.dtype)
        org_shape = x_int.shape
        if group > 0:
            x_int = x_int.reshape(x_int.shape[:-1] + (-1, group))
            scale = scale.reshape(-1, x_int.shape[-2]).unsqueeze(0).unsqueeze(
                -1)  # Unsqueeze 0 for batch dim, -1 for group dim.
            round_zero_point = round_zero_point.reshape(
                -1, x_int.shape[-2]).unsqueeze(0).unsqueeze(-1)
            if add_dim:
                scale = scale.unsqueeze(add_dim)
                round_zero_point = round_zero_point.unsqueeze(add_dim)
        if uint:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point - self.qmin)
            x_dequant = x_int.mul_(scale)
            return x_dequant.reshape(org_shape)
        else:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point)
            x_dequant = x_int.mul_(scale)
            return x_dequant.reshape(org_shape)

    def fake_quant(self, x, scale, round_zero_point):
        # start quantization
        x_int = (x / scale).round_()
        if round_zero_point is not None:
            x_int = x_int.add_(round_zero_point)
        x_int = x_int.clamp_(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub_(round_zero_point)
        x_dequant = x_dequant.mul_(scale)
        return x_dequant

    def forward(self, x: torch.Tensor):

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(255).round_().div_(255)

        # start quantization
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def free(self):
        del self.cached_xmin
        del self.cached_xmax
        del self.recorded_quant_input


class WQLinear(nn.Module):

    def __init__(self, w_bit, group_size, in_features, out_features, bias,
                 dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            'qweight',
            torch.zeros((in_features, out_features // (32 // self.w_bit)),
                        dtype=torch.int32,
                        device=dev))
        self.register_buffer(
            'qzeros',
            torch.zeros((in_features // self.group_size, out_features //
                         (32 // self.w_bit)),
                        dtype=torch.int32,
                        device=dev))
        self.register_buffer(
            'scales',
            torch.zeros((in_features // self.group_size, out_features),
                        dtype=torch.float16,
                        device=dev))
        if bias:
            self.register_buffer(
                'bias',
                torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls,
                    linear,
                    w_bit,
                    group_size,
                    init_only=False,
                    scales=None,
                    zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features,
                         linear.out_features, linear.bias is not None,
                         linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] +
                     scale_zeros[idx // group_size]) /
                    awq_linear.scales[idx // group_size]).to(torch.int)[:,
                                                                        None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=intweight.device)

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit),
            dtype=torch.int32,
            device=zeros.device)

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        out = f16s4_gemm.gemm_forward_cuda(x.reshape(-1, x.shape[-1]),
                                           self.qweight, self.scales,
                                           self.qzeros, 8)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)


class QuantLLaMaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,
        wbit: int,
        group_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        dev: str,
        dropout: float = 0.0,
        is_decoder: bool = False,
        rope_scaling=None,
        args=None,
        disable_act_quant=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.group_size = group_size  # kvcache_group_size
        self.w_bit = wbit  # kvcache bit
        self.dev = dev
        self.register_buffer(
            'kcache_qzeros',
            torch.zeros((hidden_size // self.group_size, 1),
                        dtype=torch.int32,
                        device=dev))
        self.register_buffer(
            'kcache_scales',
            torch.zeros((hidden_size // self.group_size, 1),
                        dtype=torch.float16,
                        device=dev))
        self.register_buffer(
            'vcache_qzeros',
            torch.zeros((hidden_size // self.group_size, 1),
                        dtype=torch.int32,
                        device=dev))
        self.register_buffer(
            'vcache_scales',
            torch.zeros((hidden_size // self.group_size, 1),
                        dtype=torch.float16,
                        device=dev))

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                "hidden_size must be divisible by num_heads "
                f"(got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")

        self.is_decoder = is_decoder
        self._init_rope()

        # input is quantized by LayerNorm, set disable_input_quant=True
        self.k_proj = org_module.k_proj  # org_module is after quanting.
        self.v_proj = org_module.v_proj
        self.q_proj = org_module.q_proj
        self.o_proj = org_module.o_proj

    def pack_kvcache(self, kvcache_quant_params):
        self.kcache_qzeros = kvcache_quant_params[1].to(dtype=torch.int32)
        self.kcache_scales = kvcache_quant_params[0].clone().half()
        self.vcache_qzeros = kvcache_quant_params[3].to(dtype=torch.int32)
        self.vcache_scales = kvcache_quant_params[2].clone().half()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int,
               num_x_heads: int):
        return (tensor.view(bsz, seq_len, num_x_heads,
                            self.head_dim).transpose(1, 2).contiguous())

    def _init_rope(self):
        from transformers.models.llama.modeling_llama import (
            LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding,
            LlamaDynamicNTKScalingRotaryEmbedding)
        if self.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor)
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        #  Modified from transformers/models/llama/modeling_llama.py
        from transformers.models.llama.modeling_llama import \
            (apply_rotary_pos_emb, repeat_kv)
        bsz, q_len, _ = hidden_states.size()

        # query no quant.
        query_states = self.q_proj(hidden_states)
        query_states = self._shape(query_states, -1, bsz, self.num_heads)
        # kvcache_quant
        key_states = self.k_proj(hidden_states)
        k_cache_quantizer = UniformAffineQuantizer(self.w_bit, False)
        v_cache_quantizer = UniformAffineQuantizer(self.w_bit, False)
        # key_states = self.qkt_matmul.quant_x2(key_states)
        key_states = self._shape(key_states, -1, bsz, self.num_key_value_heads)
        value_states = self.v_proj(hidden_states)
        # value_states = self.pv_matmul.quant_x2(value_states)
        value_states = self._shape(value_states, -1, bsz,
                                   self.num_key_value_heads)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        #  Need quant?
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        assert (self.head_dim % self.group_size) == 0
        quant_key_states = k_cache_quantizer.quant(key_states,
                                                   self.kcache_scales,
                                                   self.kcache_qzeros,
                                                   uint=True,
                                                   group=self.group_size,
                                                   add_dim=2)
        quant_value_states = v_cache_quantizer.quant(value_states,
                                                     self.vcache_scales,
                                                     self.vcache_qzeros,
                                                     uint=True,
                                                     group=self.group_size,
                                                     add_dim=2)
        #  Key Value states is now quantized after code if use kvcache
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], quant_key_states],
                                   dim=2)
            value_states = torch.cat([past_key_value[1], quant_value_states],
                                     dim=2)
            assert use_cache
            past_key_value = (key_states, value_states) if use_cache else None
        else:
            assert use_cache
            key_states = quant_key_states
            value_states = quant_value_states
            past_key_value = (quant_key_states,
                              quant_value_states) if use_cache else None

        del quant_key_states
        del quant_value_states
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        k_dequant = k_cache_quantizer.dequant(key_states,
                                              self.kcache_scales,
                                              self.kcache_qzeros,
                                              uint=True,
                                              group=self.group_size,
                                              add_dim=2)
        attn_weights = torch.matmul(query_states,
                                    k_dequant.transpose(2, 3)) \
            / math.sqrt(self.head_dim)
        del k_dequant
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                "Attention weights should be of size "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    "Attention mask should be of size "
                    f"{(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # Need change when kvcache quant?
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        v_dequant = v_cache_quantizer.dequant(value_states,
                                              self.vcache_scales,
                                              self.vcache_qzeros,
                                              uint=True,
                                              group=self.group_size,
                                              add_dim=2)
        attn_output = torch.matmul(attn_weights, v_dequant)
        del v_dequant
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                "`attn_output` should be of size "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
