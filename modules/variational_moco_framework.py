import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import register_model
from modules.moco_framework import DDP_MemoryQueue, ema_update, set_requires_grad

class Variational_Encoder(nn.Module):
    def __init__(self, encoder_config, **kwargs):
        from cli.builder import build_model
        super().__init__()

        # Build backbone
        self.backbone = build_model(model_config=encoder_config.get("backbone_config"))
        # self.projector = build_model(model_config=encoder_config.get("projector_config"))
        # self.fc_mu = build_model(model_config=encoder_config.get("latent_config"), out_features=encoder_config.get("latent_dim"))
        # self.fc_logvar = build_model(model_config=encoder_config.get("latent_config"), out_features=encoder_config.get("latent_dim"))


        dim_mlp_in = self.backbone.fc.weight.shape[1]
        dim_mlp_out = self.backbone.fc.weight.shape[0]
        old_fc = self.backbone.fc

        # New MLP head
        mlp = nn.Sequential(
            nn.Linear(dim_mlp_in, dim_mlp_in, True),
            nn.ReLU(),
            old_fc,                 # append original classification head
        )
        self.backbone.fc = mlp

        # --- Weight initialization ---
        for m in self.backbone.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.backbone.fc[2].weight, 0, math.sqrt(2. / dim_mlp_out))

        # Latent layers
        self.fc_mu = nn.Linear(dim_mlp_out, encoder_config.get("latent_dim"))
        self.fc_logvar = nn.Linear(dim_mlp_out, encoder_config.get("latent_dim"))

        # Initialize latent fc layers
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc_mu.bias)

        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc_logvar.bias)

        
    def reparameterize(self, mu, logvar):
        # logvar = logvar.clamp(-10, 10)
        std = (0.5 * logvar).exp()
        std = torch.clamp(std, max=100)
        eps = torch.empty_like(std).normal_()
        
        z = mu + eps * std
        
        return z

    def forward(self, x, **kwargs):
        backbone_output = self.backbone(x)
        feats = backbone_output["features"]
        representation = backbone_output["representation"]
        # proj = self.projector(feats)
        mu =  self.fc_mu(F.relu(feats)) #  self.fc_mu(F.normalize(feats, dim=1))
        logvar = self.fc_logvar(F.relu(feats))# self.fc_logvar(F.normalize(feats, dim=1))
        z = self.reparameterize(mu, logvar)

        return {"features": feats, "representation": representation, "mu": mu, "logvar":logvar, "z": z}

@register_model("variational_moco_framework")
class Variational_MoCo_Framework(nn.Module):
    """Minimal DDP-safe MoCo wrapper"""
    def __init__(self, encoder_config,K=32768, m=0.999, **kwargs):
        super().__init__()
        self.m = m

        # Encoders
        self.query_enc = Variational_Encoder(encoder_config)
        self.key_enc = Variational_Encoder(encoder_config)
        self._init_key_encoder()

        dim = encoder_config.get("latent_dim")       

        self.memory = DDP_MemoryQueue(dim=dim, K=K)

    @torch.no_grad()
    def _init_key_encoder(self):
        # Copy query weights and freeze key encoder
        for q, k in zip(self.query_enc.parameters(), self.key_enc.parameters()):
            k.data.copy_(q.data)
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
        q = out_q["z"]
        q = F.normalize(q, dim=-1)
        # q, mu_q, logvar_q = out_q["rep"], out_q["mu"], out_q["logvar"]

        # Key (no grad)
        with torch.no_grad():
            self.momentum_update_key_encoder()
            out_k = self.key_enc(x_k)
            k = out_k["z"]
            k = F.normalize(k, dim=-1)
            # k, mu_k, logvar_k = out_k["rep"], out_k["mu"], out_k["logvar"]

        # Get full memory bank
        mem_bank = self.memory.get_all()

        # Update queue
        with torch.no_grad():
            self.memory.enqueue(k)
            # self.memory.enqueue(k.detach(), mu_k.detach(), logvar_k.detach())
        
        return {
            "q": q, "k": k, "mb_negatives": mem_bank,
            "mu_q": out_q["mu"], "logvar_q": out_q["logvar"],
            "mu_k": out_k["mu"], "logvar_k": out_k["logvar"],
            # "mu_bank": mem_bank["mu_bank"], "logvar_bank": mem_bank["logvar_bank"],
            "features_q": out_q["features"], "proj_q": out_q["representation"],
            "features_k": out_k["features"], "proj_k": out_k["representation"],
        }
