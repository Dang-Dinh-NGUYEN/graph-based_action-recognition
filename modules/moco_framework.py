import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import register_model

@torch.no_grad()
def ema_update(m, q, momentum=0.999):
    """EMA update: m <- m * momentum + (1 - momentum) * q"""
    for param_m, param_q in zip(m.parameters(), q.parameters()):
        param_m.mul_(momentum).add_(param_q, alpha=(1.0 - momentum))

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather across all GPUs for DDP.
    Returns [world_size * batch_size, ...].
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    world_size = torch.distributed.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

class DDP_MemoryQueue(nn.Module):
    def __init__(self, dim, K=32000, device=None):
        super().__init__()
        self.K = K
        self.dim = dim
        self.device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        q = F.normalize(torch.randn(K, dim, device=device), dim=1)
        self.register_buffer("queue", q)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long, device=device))

         # If distributed already initialized, broadcast queue and ptr from rank 0
        # if torch.distributed.is_initialized():
        #     self.queue = self.queue.to(torch.cuda.current_device())
        #     self.ptr = self.ptr.to(torch.cuda.current_device())
        #     torch.distributed.broadcast(self.queue, src=0)
        #     torch.distributed.broadcast(self.ptr, src=0)

    @torch.no_grad()
    def enqueue(self, reps):
        """
        Enqueue reps from all GPUs into the global circular queue.
        reps: [B, D]
        """
        reps_all = concat_all_gather(reps)  # [B_total, D]
        reps_all = F.normalize(reps_all, dim=1)
        
        B_total = reps_all.size(0)
        K = self.K
        ptr = int(self.ptr.item())

        # Circular update
        if ptr + B_total <= K:
            self.queue[ptr:ptr + B_total] = reps_all
        else:
            first = K - ptr
            self.queue[ptr:] = reps_all[:first]
            self.queue[:B_total - first] = reps_all[first:]

        self.ptr[0] = (ptr + B_total) % K

    @torch.no_grad()
    def get_all(self, device=None):
        device = device if device is not None else self.device
        return self.queue.clone().detach().to(device)
     
class Encoder(nn.Module):
    def __init__(self, encoder_config, **kwargs):
        from cli.builder import build_model
        super().__init__()
        self.backbone = build_model(model_config=encoder_config.get("backbone_config"))
        dim_mlp = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                            nn.ReLU(),
                                            self.backbone.fc)

        # self.projector = build_model(model_config=encoder_config.get("projector_config"))

    def forward(self, x, **kwargs):
        feats = self.backbone(x)["features"]
        return {"features": feats, "proj": feats}

@register_model("moco_framework")
class MoCo_Framework(nn.Module):
    """Minimal DDP-safe MoCo wrapper"""
    def __init__(self, encoder_config, K=32768, m=0.999, **kwargs):
        super().__init__()
        self.m = m

        # Encoders
        self.query_enc = Encoder(encoder_config)
        self.key_enc = Encoder(encoder_config)
        self._init_key_encoder()

        dim = self.query_enc.backbone.fc[-1].out_features

        self.memory = DDP_MemoryQueue(dim=dim, K=K)

    @torch.no_grad()
    def _init_key_encoder(self):
        # Copy query weights and freeze key encoder
        self.key_enc.load_state_dict(self.query_enc.state_dict())
        set_requires_grad(self.key_enc, False)
        # self.key_enc.eval()

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        ema_update(self.key_enc, self.query_enc, momentum=self.m)

    def forward(self, batch, **kwargs):
        if isinstance(batch, torch.Tensor):
            return self.query_enc(batch)

        x_q = batch["anchor"]
        x_k = batch["positive"]

        # Query (with grad)
        out_q = self.query_enc(x_q)
        q = out_q["proj"]
        q = F.normalize(q, dim=1)

        # Key (no grad)
        with torch.no_grad():
            self.momentum_update_key_encoder()
            out_k = self.key_enc(x_k)
            k = out_k["proj"]
            k = F.normalize(k, dim=1)

        # Get full memory bank
        mem_bank = self.memory.get_all()

        # Update queue
        self.memory.enqueue(k)

        return {
            "q": q, "k": k, "mb_negatives": mem_bank,
            "features_q": out_q["features"], "proj_q": out_q["proj"],
            "features_k": out_k["features"], "proj_k": out_k["proj"],
        }
