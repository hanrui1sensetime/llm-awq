import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval",
                      tokenizer=None,
                      n_samples=512,
                      block_size=512):
    if data == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train")
    if data == "custom":
        dataset = load_dataset(
            "json",
            data_files="/root/workspace/external_data/llama_calib.jsonl",
            split="train")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        if 'input_ids' in data:
            line_encoded = data['input_ids']
        else:
            line = data["text"]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size:(i + 1) * block_size]
        for i in range(n_split)
    ]
