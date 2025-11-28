from collections.abc import Mapping
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from cli.ddp_utils import set_random_seed
from data_generator import recursive_stack, recursive_to_device, register_batch_handler, register_feeder
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler

@register_feeder("basic_dataset")
class BasicDataset(Dataset):
    def __init__(self, base_dataset, cached_features: dict = None):
        self.base = base_dataset
        self.cached = cached_features
      
    def __len__(self): 
        return len(self.base)

    def __getitem__(self, idx):
        if isinstance(idx, list):  # batched access
            return [self.__getitem__(i) for i in idx]

        if self.cached is None:
            if hasattr(self.base[idx], 'id'):
                return (self.base[idx].x, self.base[idx].y, self.base[idx].id)
            return (self.base[idx].x, self.base[idx].y)
        else:
            return (self.cached[idx], self.base[idx].y, self.base[idx].id)


class DistributedBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, rank, world_size,
                 drop_last=False, shuffle=False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 0

        # make number of samples evenly divisible
        self.num_samples_per_rank = int(math.ceil(dataset_size / world_size))
        self.total_size = self.num_samples_per_rank * world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 1. generate original indices
        indices = list(range(self.dataset_size))

        # 2. shuffle
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()

        # 3. pad
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]

        # 4. split contiguously
        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        indices = indices[start:end]

        # 5. chunk into batches
        batches = [
            indices[i:i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]

        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return self.num_samples_per_rank // self.batch_size
        else:
            return (self.num_samples_per_rank + self.batch_size - 1) // self.batch_size

@register_batch_handler("basic_batch_handler")
class BasicBatchHandler:
    def __init__(self, config, dataset_raw, rank=0, world_size=1, shuffle=False):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.dataset = dataset_raw
        self.batch_sampler = None

    def build_dataloader(self):
        batch_sampler = DistributedBatchSampler(len(self.dataset), batch_size=self.config['batch_size'], rank=self.rank, world_size=self.world_size, shuffle=self.shuffle, drop_last=False)

        # shuffle = False if sampler else self.config.get('shuffle', True)
        return DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            num_workers=8,
            # collate_fn=self.collate_fn 
            # worker_init_fn=set_random_seed
        )
    
    def collate_fn(self, batch):
        """Custom collate_fn to handle dict features or raw tensors."""
        # inputs, labels, ids = zip(*batch)

        inputs, labels = zip(*batch)
    
        inputs_batch = recursive_stack(inputs)
        label_batch = torch.stack(labels)

        return inputs_batch, label_batch # , ids
    
    def load_batch(self, batch, device):
        if len(batch) == 3:
            inputs, labels, ids = batch
        else:
            inputs, labels = batch
            ids = None

        labels = labels - 1
      
        # Detect whether we got raw tensors or cached‐feature dicts
        if isinstance(inputs, Mapping):
            # cached‐feature mode
            frozen = {
                "frozen_inputs": recursive_to_device(inputs, device),
            }
            inputs = None
        else:
            # raw‐tensor mode
            inputs = {
                "raw_inputs": inputs.squeeze(1).to(device, non_blocking=True),
            }
            frozen = None

        return {
            "inputs": inputs,
            "frozen_features": frozen,
            "labels": labels.to(device, non_blocking=True),
            "ids": ids
        }