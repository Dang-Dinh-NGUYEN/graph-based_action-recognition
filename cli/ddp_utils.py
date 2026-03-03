import os
import random
import numpy as np
import torch
import torch.distributed as dist

def setup_ddp(rank, world_size, master_addr="localhost", master_port="9991"):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def set_random_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_ddp():
    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()