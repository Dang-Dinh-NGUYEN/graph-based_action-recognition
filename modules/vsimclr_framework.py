import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import register_model
from modules.moco_framework import DDP_MemoryQueue, ema_update, set_requires_grad
from modules.variational_moco_framework import Variational_Encoder

@register_model("vsimclr_framework")
class Variational_SimCLR_Framework(nn.Module):
    """Minimal DDP-safe SimCLR wrapper"""
    def __init__(self, encoder_config,**kwargs):
        super().__init__()

        # Encoders
        self.query_enc = Variational_Encoder(encoder_config)

    def forward(self, batch, **kwargs):
        if isinstance(batch, torch.Tensor):
            return self.query_enc(batch)

        x_q = batch["anchor"]
        x_k = batch["positive"]

        # Query (with grad)
        out_q = self.query_enc(x_q)
        q = out_q["z"]
        q = F.normalize(q, dim=-1)
        # q, mu_q, logvar_q = out_q["rep"], out_q["mu"], out_q["logvar"]

        out_k = self.query_enc(x_k)
        k = out_k["z"]
        k = F.normalize(k, dim=-1)
        # k, mu_k, logvar_k = out_k["rep"], out_k["mu"], out_k["logvar"]

        return {
            "q": q, "k": k, "mb_negatives": None,
            "mu_q": out_q["mu"], "logvar_q": out_q["logvar"],
            "mu_k": out_k["mu"], "logvar_k": out_k["logvar"],
            # "mu_bank": mem_bank["mu_bank"], "logvar_bank": mem_bank["logvar_bank"],
            "features_q": out_q["features"], "proj_q": out_q["representation"],
            "features_k": out_k["features"], "proj_k": out_k["representation"],
        }
