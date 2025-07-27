import math
from typing import List, Union, Optional

import torch
import numpy as np

from lm_experiments_tools.utils import get_distributed_rank
from tqdm.auto import tqdm


class MixtureDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset],
                 weights: Optional[List[Union[float, int]]] = None) -> None:
        """MixtureDataset takes each dataset from datasets list with its weight.

        datasets = [d1, d2, d3]
        weights = [1.0, 2.0, 0.5]
        -> len(MixtureDataset) = 1.0 * len(d1) + 2.0 * len(d2) + 0.5 * len(d3)
           MixtureDataset = d1.sample(1.0 * len(d1)) + d2.sample(2.0 * len(d2)) + d3.sample(0.5 * len(d3)),
           where d.sample(n) takes n samples from dataset d

        MixtureDataset is similar to megatron.data.blendable_dataset.BlendableDataset, but has different
        blending/mixturing logic:
            len(BlendableDataset) = len(d1) + len(d2) + len(d3) = len(d)
            weights /= sum(weights)
            BlendableDataset =  d1.sample(w1 * len(d)) + d2.sample(w2 * len(d)) + d3.sample(w3 * len(d)),
            resulting in under-/up-sampling from d1, d2, d3.

        Args:
            datasets (List[torch.utils.data.Dataset]): list of torch Datasets
            weights (Optional[List[Union[float, int]]], optional): weights of datasets, if weights is None
                than we just merge all datasets into one. Defaults to None.
        """
        if weights is None:
            weights = [1.0] * len(datasets)

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        self.num_samples = []
        for w, dataset in zip(weights, self.datasets):
            self.num_samples += [math.ceil(w * len(dataset))]
        self.size = np.sum(self.num_samples)

        weights = np.array(self.num_samples, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        from megatron.data import helpers
        helpers.build_blending_indices(self.dataset_index,
                                       self.dataset_sample_index,
                                       weights, num_datasets, self.size,
                                       get_distributed_rank() == 0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx]


def generate_ar_pairs(key_size, value_size, num_pairs, num_samples, rewrite_setting=False, num_symbols=None):
    keys = torch.empty((num_samples, num_pairs, key_size))

    if not rewrite_setting:
        for i in tqdm(range(num_samples)):
            key = torch.randperm(num_symbols ** key_size)[:num_pairs]
            for j in range(key_size):
                keys[i, :, j] = key % num_symbols
                key //= num_symbols
    else:
        keys = torch.randint(0, num_symbols, (num_samples, num_pairs, key_size))
    
    values = torch.randint(0, num_symbols, (num_samples, num_pairs, value_size))

    return keys, values


class ARDataset:
    def __init__(self, key_size, value_size, sample_len=1, num_samples=20_000, rewrite_setting=False, num_symbols=16):
        self.sample_len = sample_len
        self.keys, self.values = generate_ar_pairs(key_size, value_size, sample_len, num_samples, rewrite_setting, num_symbols)

        if not rewrite_setting:
            self.target_key_inds = torch.randint(sample_len, (num_samples, ))
        else:
            self.target_key_inds = torch.empty((num_samples,), dtype=torch.long)
            for i in tqdm(range(num_samples)):
                unique_keys = self.keys[i].unique(dim=0)
                key = unique_keys[torch.randperm(len(unique_keys))[0]]
                try:
                    idx = torch.max(torch.where(torch.all(self.keys[i] == key, dim=-1))[0], dim=0)[0].long()
                except Exception:
                    print(f"{self.keys[i]}, {key}")
                    raise 1
                assert torch.all(self.keys[i][idx] == key)
                self.target_key_inds[i] = idx

    def __getitem__(self, idx):
        keys, values, tgt_ind = self.keys[idx], self.values[idx], self.target_key_inds[idx]
        sample = {'keys': keys, 'values': values, 'target_key_ind': tgt_ind}
        return sample
    
    def __len__(self):
        return self.keys.shape[0]


def ar_collate_fn(batch, valid=False, vary_n_segments=False, rewrite_setting=False, sep_token=None, eos_token=None, gen_token=None, value_size=1):
    keys = [b['keys'] for b in batch]
    values = [b['values'] for b in batch]
    
    if not vary_n_segments:
        tgt_inds = [b['target_key_ind'].item() for b in batch]
        n = len(keys[0])
    else:
        n = torch.randint(1, len(keys[0])+1, size=())
        keys = [x[-n:] for x in keys]
        values = [x[-n:] for x in values]
        if not rewrite_setting:
            tgt_inds = [torch.randint(0, n, size=()).item() for _ in range(len(keys))]
        else:
            tgt_inds = []
            for i in range(len(keys)):
                unique_keys = keys[i].unique(dim=0)
                key = unique_keys[torch.randperm(len(unique_keys))[0]]
                try:
                    idx = torch.max(torch.where(torch.all(keys[i] == key, dim=-1))[0], dim=0)[0].long()
                except Exception:
                    print(f"{keys[i]}, {key}")
                    raise 1
                assert torch.all(keys[i][idx] == key)
                tgt_inds.append(idx)

    bs = len(keys)
    sep_tokens = torch.ones(bs, 1) * sep_token
    eos_tokens = torch.ones(bs, 1) * eos_token
    gen_tokens = torch.ones(bs, 1) * gen_token
    sample = []

    for i in range(n):
        sample.append(torch.stack([k[i] for k in keys]))
        sample.append(sep_tokens)
        sample.append(torch.stack([v[i] for v in values]))
        sample.append(eos_tokens)

    target_keys = torch.stack([k[i] for i, k in zip(tgt_inds, keys)])
    target_values = torch.stack([k[i] for i, k in zip(tgt_inds, values)])

    sample.append(target_keys)
    sample.append(gen_tokens)

    input_ids_generate = torch.cat(sample, dim=1)

    sample.append(target_values)
    sample.append(eos_tokens)
    input_ids = torch.cat(sample, dim=1)

    labels_mask = torch.zeros_like(input_ids).bool()
    labels_mask[:, -value_size - 2:] = True

    collated = {'input_ids': input_ids.long(), 
                'input_ids_generate': input_ids_generate.long(), 
                'attention_mask': torch.ones_like(input_ids).bool(),
                'attention_mask_generate': torch.ones_like(input_ids_generate).bool(),
                'labels': input_ids.long(), 
                'labels_mask': labels_mask, 
                }
    return collated

