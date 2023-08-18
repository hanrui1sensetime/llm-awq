import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip

__all__ = ["run_awq"]


def get_named_linears(module):
    return {
        name: m
        for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = \
            model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = \
            model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = \
            model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = \
            model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = \
            model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    use_kvcache=False,
    kvcache_bit=8,
    kvcache_groupsize=128,
    # some configs for ablation study
    calib_data="pileval",
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    layers = get_blocks(model)

    samples = get_calib_dataset(data=calib_data,
                                tokenizer=enc,
                                n_samples=n_samples,
                                block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {"scale": [], "clip": [], "kvcache": []}

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict, kv_cache=False):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)
            if kv_cache and ('k_proj' in name or 'v_proj' in name):
                if 'k_proj' in name:
                    if 'k_cache' not in feat_dict:
                        feat_dict['k_cache'] = [y]
                    else:
                        feat_dict['k_cache'].append(y)
                else:
                    if 'v_cache' not in feat_dict:
                        feat_dict['v_cache'] = [y]
                    else:
                        feat_dict['v_cache'].append(y)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook,
                                  name=name,
                                  feat_dict=input_feat,
                                  kv_cache=True)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        if use_kvcache:
            kv_cache_feat = dict(k_cache=input_feat['k_cache'],
                                 v_cache=input_feat['v_cache'])
            input_feat.pop('k_cache')
            input_feat.pop('v_cache')
            batch = kv_cache_feat['k_cache'].shape[0]
            length = kv_cache_feat['k_cache'].shape[1]
            from .quantizer import pseudo_quantize_tensor
            _, k_cache_scale, k_cache_zp = pseudo_quantize_tensor(
                kv_cache_feat['k_cache'].reshape(
                    batch, length, -1,
                    kvcache_groupsize).permute(2, 0, 1, 3).reshape(
                        -1, batch * length * kvcache_groupsize),
                kvcache_bit,
                True,
                batch * length * kvcache_groupsize,
                get_scale_zp=True)
            _, v_cache_scale, v_cache_zp = pseudo_quantize_tensor(
                kv_cache_feat['v_cache'].reshape(
                    batch, length, -1,
                    kvcache_groupsize).permute(2, 0, 1, 3).reshape(
                        -1, batch * length * kvcache_groupsize),
                kvcache_bit,
                True,
                batch * length * kvcache_groupsize,
                get_scale_zp=True)
            awq_results['kvcache'].append(
                (kvcache_bit, kvcache_groupsize, k_cache_scale, k_cache_zp,
                 v_cache_scale, v_cache_zp))

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list,
                get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list,
                get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


def apply_kvcache(model, kvcache_bit, kvcache_group_size, awq_kvcache):
    from .qmodule import QuantLLaMaAttention
    from .quantizer import set_op_by_name
    layers = get_blocks(model)
    for i in tqdm.tqdm(range(len(layers)),
                       desc="apply kvcache for LLaMa Attention"):
        layer = layers[i]
        kvcache_bit = awq_kvcache[i][0]
        kvcache_group_size = awq_kvcache[i][1]
        kvcache_quant_param = awq_kvcache[i][2:]
        quant_llama_attention = QuantLLaMaAttention(
            layer.self_attn,
            kvcache_bit,
            kvcache_group_size,
            layer.self_attn.max_position_embeddings,
            layer.self_attn.hidden_size,
            layer.self_attn.num_heads,
            layer.self_attn.num_key_value_heads,
            layer.self_attn.k_proj.weight.device,
            rope_scaling=layer.self_attn.config.rope_scaling)
        set_op_by_name(layer, "self_attn", quant_llama_attention)
        layer.self_attn.pack_kvcache(kvcache_quant_param)


def apply_awq(model, awq_results, use_kvcache=False):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])

    if use_kvcache:
        apply_kvcache(model, awq_results["kvcache"][0][0],
                      awq_results["kvcache"][0][1], awq_results["kvcache"])
