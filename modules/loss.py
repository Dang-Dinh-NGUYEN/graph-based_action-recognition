import math
import torch
import torch.nn as nn
from modules import register_loss
from abc import ABC, abstractmethod
import torch.nn.functional as F
import modules
from torch import linalg as LA
from torch.nn.parameter import Parameter
from torch.nn import init

def _safe_check(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[NaN/Inf detected in {name}] min={tensor.min().item()}, max={tensor.max().item()}")
    return tensor

class LossStrategy(ABC):
    @abstractmethod
    def compute_loss(self, model_output : dict, labels : torch.tensor, step: int = None, epoch: int = None) -> dict:
        pass

@register_loss("bce")
class BCELossStrategy(LossStrategy):
    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def compute_loss(self, model_output, labels, **kwargs):
        logits = model_output['logits']       # keep as float [N, 1]
        labels = labels.float().to(logits.device)  # ensure float and same device
        bce_loss = self.criterion(logits, labels)
        return {"loss": bce_loss, "clf_loss": bce_loss}
    
@register_loss("clf")
class ClfLossStrategy(LossStrategy):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, model_output, labels, **kwargs):
        logits = model_output['logits']  # [N, num_classes]
        # Ensure labels are 1-D and on the same device, integer type
        labels = labels.squeeze(-1).to(logits.device)  # [N]
        clf_loss = self.criterion(logits, labels)

        return {"loss": clf_loss, "clf_loss": clf_loss}


@register_loss("soft_triple_loss")
class SoftTripleLoss(LossStrategy):
    def __init__(self, num_classes=60, embedding_dim=256, centers_per_class=5, tau=0.1, margin=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.centers_per_class = centers_per_class
        self.tau = tau
        self.margin = margin

        # Total centers = num_classes * centers_per_class
        self.centers = nn.Parameter(torch.randn(num_classes * centers_per_class, embedding_dim))
        nn.init.kaiming_normal_(self.centers)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, model_output, labels, **kwargs):
        features = F.normalize(model_output["features"], p=2, dim=-1)  # [B, D]
        centers = F.normalize(self.centers, p=2, dim=-1).to(features.device)  # [C*K, D]

        # Compute similarity to all centers
        sim_all = torch.matmul(features, centers.T)  # [B, C*K]
        sim_all = sim_all.view(-1, self.num_classes, self.centers_per_class)  # [B, C, K]

        # Soft assignment over centers per class
        soft_assign = F.softmax(sim_all / self.tau, dim=2)  # [B, C, K]
        weighted_sim = (soft_assign * sim_all).sum(dim=2)  # [B, C]

        logits = weighted_sim - self.margin  # Optionally subtract margin

        loss = self.criterion(logits, labels.squeeze())
        return {
            "loss": loss,
            "soft_triple_loss": loss
        }
"""

@register_loss("soft_triple_loss")
class SoftTripleLoss(nn.Module):
    def __init__(self, la=20, gamma=0.1, tau=0, margin=0.0001, dim=256, cN=60, K=1):
        super(SoftTripleLoss, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, self.cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=bool)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, input, target, **kwargs):
        input=input["features"]
        centers = F.normalize(self.fc, p=2, dim=0).to(input.device)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
        
    def compute_loss(self, input, target, **kwargs):
        loss = self.forward(input, target, **kwargs)
        return {"loss": loss, "soft_triple_loss": loss}
"""

@register_loss("kl_loss")
class KLLossStrategy(LossStrategy):
    def __init__(self, start_weight=0.0, end_weight=1.0, anneal_by='epoch', anneal_duration=20):
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.anneal_by = anneal_by
        self.anneal_duration = anneal_duration

    def _get_weight(self, epoch=None, step=None):
        # Select annealing variable
        x = epoch if self.anneal_by == 'epoch' else step
        if x is None:
            return self.end_weight
        progress = min(max(x / self.anneal_duration, 0.0), 1.0)
        return self.start_weight + (self.end_weight - self.start_weight) * progress

    def compute_loss(self, model_output, labels, epoch=None, step=None, **kwargs):
        weight = self._get_weight(epoch=epoch, step=step)

        def _compute(mu, logvar):
            mu = torch.clamp(mu, -10, 10)
            logvar = torch.clamp(logvar, -10, 10)
            kl_per_sample = -0.5 * torch.mean(-mu.pow(2) - logvar.exp() + 1 + logvar)# .sum(dim=1)
            kl = kl_per_sample

            # Optional: gradient clipping
            # if kl.requires_grad:
                # kl.register_hook(lambda g: torch.clamp(g, -0.5, 0.5))

            return weight * kl, kl

        w_anchor_kl, anchor_kl = _compute(model_output['mu_q'], model_output['logvar_q'])
        # p_anchor_kl, positive_kl = _compute(model_output['mu_k'], model_output['logvar_k'])
        kl_loss = anchor_kl  # + positive_kl
        weighted_kl = w_anchor_kl  # + p_anchor_kl

        return {"loss": weighted_kl, "kl_loss": kl_loss, "kl_weight": weight}
    
@register_loss("triplet_loss")
class TripletLossStrategy(LossStrategy):
    def __init__(self, margin=0.07):
        self.margin = margin
        self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')

    def compute_loss(self, model_output, labels, **kwargs):
        """
        v1 = model_output["z_anchor"]
        v2 = model_output["z_positive"]
        v3 = model_output["z_negative"]

        v1 = F.normalize(v1, dim=-1)
        v2 = F.normalize(v2, dim=-1)
        v3 = F.normalize(v3, dim=-1)
        
        scores = torch.matmul(v1, v2.permute(*range(v2.dim() - 2), -1, -2))

        class_size = scores.size(-2)
        positive = scores.diagonal(dim1=-2, dim2=-1)
        negative_without_positive = scores - (
            torch.eye(class_size, device=scores.device) * 2.0
        )
        closest_negative, _ = torch.max(negative_without_positive, dim=-1)
        negative_zero_on_duplicate = scores * (
            1.0 - torch.eye(class_size, device=scores.device)
        )
        mean_negative = torch.sum(negative_zero_on_duplicate, dim=-1) / (class_size - 1)
        triplet_loss1 = torch.clamp(self.margin + closest_negative - positive, min=0.0)
        triplet_loss2 = torch.clamp(self.margin + mean_negative - positive, min=0.0)
        triplet_loss = torch.mean(triplet_loss1 + triplet_loss2)

        return {"loss":triplet_loss, "triplet_loss": triplet_loss}
        """
        z_anchor   = F.normalize(model_output["decoded"], dim=-1)
        z_positive = F.normalize(model_output["decoded_positive"], dim=-1)
        z_negative = F.normalize(model_output["decoded_negative"], dim=-1)

        triplet_loss = self.criterion(z_anchor, z_positive, z_negative)
        if torch.isnan(triplet_loss):
            print("Triplet loss is NaN")
        return {
            "loss": triplet_loss,
            "triplet_loss": triplet_loss
        }
    
@register_loss("soft_margin_triplet_loss")
class SoftMarginTripletLossStrategy(LossStrategy):
    def compute_loss(self, model_output, labels, **kwargs):
        z_anchor   = F.normalize(model_output["decoded"], dim=-1)
        z_positive = F.normalize(model_output["decoded_positive"], dim=-1)
        z_negative = F.normalize(model_output["decoded_negative"], dim=-1)

        d_ap = F.pairwise_distance(z_anchor, z_positive, p=2)
        d_an = F.pairwise_distance(z_positive, z_negative, p=2)
        
        # Soft margin loss
        loss = torch.log1p(torch.exp(d_ap - d_an)).mean()
        return {
            "loss": loss,
            "soft_margin_triplet_loss": loss
        }

@register_loss("info_nce_loss")
class InfoNCELossStrategy(LossStrategy):
    def __init__(self, tau=0.07, knn_pos_K=0, topk=0, normalize=True):
        super().__init__()
        self.tau = tau
        self.knn_pos_K = knn_pos_K     # Use kNN positives from memory
        self.topk = topk               # Optional: hard negative mining
        self.normalize = normalize
        self.criterion = nn.CrossEntropyLoss()

    def _normalize_if(self, *tensors):
        if not self.normalize:
            return tensors
        return tuple(F.normalize(t, dim=-1) for t in tensors)
    
    def _simclr_batch(self, z1, z2):
        z1, z2 = self._normalize_if(z1, z2)
        B = z1.size(0)
        Z = torch.cat([z1, z2], dim=0)                 # [2B,D]
        S = (Z @ Z.t()) / self.tau                     # [2B,2B]
        eye = torch.eye(2*B, device=Z.device, dtype=torch.bool)
        S = S.masked_fill(eye, float('-inf'))          # exclude self-sim exactly
        pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(Z.device)
        pos_logits = S[torch.arange(2*B, device=Z.device), pos]
        denom = torch.logsumexp(S, dim=1)
        return -(pos_logits - denom).mean()

    def _knn_pos_logits(self, q, memory_bank):
        # q: [B, D], memory_bank: [M, D]
        sim = q @ memory_bank.to(q.dtype).T                     # [B, M]
        topk_sim, topk_idx = sim.topk(self.knn_pos_K, dim=1)
        # log-sum-exp over top-k similarities (numerator)
        knn_logits = torch.logsumexp(topk_sim / self.tau, dim=1, keepdim=True)  # [B, 1]
        return knn_logits

    def _with_knn_positives(self, q, k, memory_bank):
        q, k, memory_bank = self._normalize_if(q, k, memory_bank)
        B = q.size(0)

        # Original positive logits
        pos_logits = (q * k).sum(dim=1, keepdim=True)  # [B, 1]

        # Optional: include knn positives from memory
        knn_logits = self._knn_pos_logits(q, memory_bank)         # [B, 1]
        # Combine numerator (pos + knn)
        combined_numerator = torch.logsumexp(
            torch.cat([pos_logits, knn_logits], dim=1), dim=1
        ).unsqueeze(1)  # [B, 1]

        # Negatives
        if memory_bank.numel() == 0:
            # fallback: no negatives, just use numerator
            print("empty queue")
            logits = combined_numerator
            labels = torch.zeros(B, dtype=torch.long, device=q.device)
            loss = F.cross_entropy(torch.cat([combined_numerator,
                                            combined_numerator.detach()], dim=1), labels)
        else:
            neg_logits = q @ memory_bank.to(q.dtype).T   # [B, M]

            # ---- top-k mining (cross-swap)
            if self.topk > 0:
                _, topk_idx = torch.topk(neg_logits, self.topk, dim=1)            # [B,topk]

                B, M = neg_logits.shape
                rows = torch.arange(B, device=q.device).unsqueeze(1)

                topk_mask = torch.zeros_like(neg_logits).scatter(1, topk_idx, 1)

                # mark cross-topk as additional positives
                pos_logits = torch.cat([pos_logits, (neg_logits * topk_mask).sum(dim=1, keepdim=True)], dim=1)

        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.tau  # [B, 1+M]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = self.criterion(logits, labels)

        return loss
    
    def compute_loss(self, model_output, labels, step=None, epoch=None, **kwargs):
        if all(k in model_output for k in ["q", "k", "mb_negatives"]):
            # for k, v in model_output.items():
                #a = _safe_check(v, k)
            loss = self._with_knn_positives(
                model_output["q"],
                model_output["k"],
                model_output["mb_negatives"]
            )
        else:
            # fallback to SimCLR-style loss
            loss = self._simclr_batch(model_output["q"], model_output["k"])

        return {"loss": loss, "info_nce_loss": loss}

@register_loss("simple_info_nce_loss")
class SimpleInfoNCELossStrategy(LossStrategy):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, model_output, labels=None, **kwargs):
        # q = F.normalize(model_output["q"], dim=1)
        # k = F.normalize(model_output["k"], dim=1)
        q = model_output["q"]
        k = model_output["k"]
        mb_negatives = model_output.get("mb_negatives", None)
        # print(mb_negatives.shape)

        if mb_negatives is not None:  # --- MoCo-style ---
            l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)      # [N,1]
            l_neg = torch.einsum("nc,kc->nk", q, mb_negatives)        # [N,K]
            logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)
            loss = self.criterion(logits, labels)

        else:  # --- SimCLR-style ---
            z = torch.cat([q, k], dim=0)  # [2N,D]
            N = q.size(0)
            sim = torch.matmul(z, z.T) / self.tau
            sim.fill_diagonal_(-float("inf"))
            targets = torch.arange(N, device=z.device)
            targets = torch.cat([targets + N, targets], dim=0)
            loss = self.criterion(sim, targets)

        return {"loss": loss, "info_nce_loss": loss}
        
@register_loss("beta_nt_xent_loss")
class BetaNTXentMoCoLoss(nn.Module):
    def __init__(self, temperature=0.1, sigma=0.5, beta=0.5, dim=128):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma
        self.beta = beta
        self.dim = dim

        # Precompute constants
        self.const1 = (1 + beta) / beta
        self.const2 = 1.0 / ((2 * math.pi * (sigma ** 2)) ** (beta/2))

    def beta_gaussian_similarity(self, z):
        """
        Compute β-Gaussian similarity matrix, faithful to the first version
        but using fast squared-L2 computation.
        """
        # Normalize (optional, improves stability)
        # z = F.normalize(z, dim=1)

        # Squared pairwise distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
        norm = (z ** 2).sum(dim=1, keepdim=True)
        d2 = norm + norm.T - 2 * (z @ z.T)
        d2 = torch.clamp(d2, min=0.0)

        # β-Gaussian kernel
        term = torch.exp(-self.beta * d2 / (2 * self.sigma ** 2))
        sim = self.const1 * (self.const2 * term - 1.0)
        
        return sim

    def compute_loss(self, model_output, labels=None, step=None, epoch=None, **kwargs):
        zis, zjs = model_output["q"], model_output["k"]
        device = zis.device
        B = zis.size(0)

        # Normalize for numerical stability
        zis, zjs = F.normalize(zis, dim=1), F.normalize(zjs, dim=1)
        z = torch.cat([zis, zjs], dim=0)  # [2B, D]

        # Compute β-Gaussian similarity matrix
        sim = self.beta_gaussian_similarity(z) / self.temperature

        # Positive pairs: i↔i+B
        pos = torch.arange(B, device=device)
        sim_pos = torch.cat([sim[pos, pos + B], sim[pos + B, pos]], dim=0)

        # Mask self-similarities to exclude them from negatives
        sim.fill_diagonal_(-float("inf"))

        # Compute InfoNCE-style loss
        sim_neg = torch.logsumexp(sim, dim=1)
        loss = (sim_neg - sim_pos).mean()

        return {"loss": loss, "beta_nt_xent_loss": loss}

    
@register_loss("beta_distribution_similarity_loss")
class DistributionSimilarityLoss(LossStrategy):
    """
    Distribution similarity loss (approximated Jensen-Shannon) between two Gaussian distributions q_i and q_j.

    Formula:
        l_ij_dist = 0.5 * [(log σ_m - log σ_i) + (log σ_m - log σ_j)]
                  + 0.25 * [(μ_i - μ_m)^2 + (μ_j - μ_m)^2] / σ_m^2

    Supports annealing via start_weight / end_weight.
    """

    def __init__(self, start_weight=0.0, end_weight=1.0, anneal_by='epoch', anneal_duration=20, eps=1e-2):
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.anneal_by = anneal_by
        self.anneal_duration = anneal_duration
        self.eps = eps

    def _get_weight(self, epoch=None, step=None):
        x = epoch if self.anneal_by == 'epoch' else step
        if x is None:
            return self.end_weight
        progress = min(max(x / self.anneal_duration, 0.0), 1.0)
        return self.start_weight + (self.end_weight - self.start_weight) * progress
    
    def _l1(self, x, y):
        return x - y
    
    def _l2(self, x, y):
        return (x - y)**2

    def compute_loss(self, model_output, labels=None, epoch=None, step=None, **kwargs):
        mu_q, logv_q = model_output['mu_q'], model_output['logvar_q']
        mu_k, logv_k = model_output['mu_k'], model_output['logvar_k']

        if epoch < 150:
            mu_q = mu_q.detach()
            logv_q = logv_q.detach()
            mu_k = mu_k.detach()
            logv_k = logv_k.detach()

        # Clamp log-variance BEFORE exponentiation
        # logv_q = torch.clamp(logv_q, min=-10, max=10)
        # logv_k = torch.clamp(logv_k, min=-10, max=10)

        # Compute variances safely
        var_q = torch.exp(logv_q)
        var_k = torch.exp(logv_k)

        # --- Stable mixture ---
        mu_m = 0.5 * (mu_q + mu_k)
        var_m = 0.5 * (var_q + var_k)

        # Prevent log(0) or divide-by-zero
        # var_m = torch.clamp(var_m, min=1e-6, max=1e6)
        # var_q = torch.clamp(var_q, min=1e-6, max=1e6)
        # var_k = torch.clamp(var_k, min=1e-6, max=1e6)

        log_var_m = torch.log(var_m)

        # --- Compute terms safely ---
        ft = (log_var_m - logv_q)**2 + (log_var_m - logv_k)**2
        st = ((mu_q - mu_m) ** 2 + (mu_k - mu_m) ** 2) / (2 * var_m)

        loss = 0.5 * (ft + st).mean()
        # loss = torch.mean(torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))

        weight = self._get_weight(epoch=epoch, step=step)
        weighted_loss = weight * loss

        return {
            "loss": weighted_loss,
            "distribution_similarity_loss": loss,
            "weight": weight
        }
    
@register_loss("mmd_loss")    
class MMDLoss(LossStrategy):
    def __init__(self, lambda_1, lambda_2, num_class=60, start_weight=0.0, end_weight=1.0, anneal_by='epoch', anneal_duration=20):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.num_class = num_class
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.anneal_by = anneal_by
        self.anneal_duration = anneal_duration
        self.z_prior = torch.empty(num_class, 60)
        nn.init.orthogonal_(self.z_prior, gain=1)

    def _get_weight(self, epoch=None, step=None):
        # Select annealing variable
        x = epoch if self.anneal_by == 'epoch' else step
        if x is None:
            return self.end_weight
        progress = min(max(x / self.anneal_duration, 0.0), 1.0)
        return self.start_weight + (self.end_weight - self.start_weight) * progress

    def compute_loss(self, model_output, labels, epoch=None, step=None, **kwargs):
        num_classes = model_output["logits"].shape[-1]
        labels = labels.view(-1)
        # List of valid labels present in current batch
        valid_label = [i_cls in labels.tolist() for i_cls in range(num_classes)]

        # Compute per-class latent means
        z = model_output["features"]  # Shape: (N, D)
        
        z_mean = torch.stack([
            z[labels == i_cls].mean(dim=0) if (labels == i_cls).sum() > 0 else torch.zeros_like(z[0])
            for i_cls in range(self.num_class)
        ], dim=0).to(z.device)  # Shape: (num_classes, D)
        

        # L2 norm of overall mean
        l2_z_mean = torch.linalg.norm(z.mean(dim=0), ord=2)  # Scalar
        # Apply valid_label mask
        valid_label_mask = torch.tensor(valid_label, dtype=torch.bool, device=model_output["features"].device)

        z_mean_valid = z_mean[valid_label_mask]                     # Shape: (K, D)
        self.z_prior = self.z_prior.to(z_mean_valid.device)
        z_prior_valid = self.z_prior[valid_label_mask].to(z_mean_valid.device)   # Shape: (K, D)
        # MMD loss (using MSE)
        mmd_loss = F.mse_loss(z_mean_valid, z_prior_valid)

        # Final loss
        loss = self.lambda_1 * l2_z_mean + self.lambda_2 * mmd_loss
        
        weight = self._get_weight(epoch=epoch, step=step)
        weighted_mmd = weight * loss

        return {"loss": weighted_mmd, "mmd_loss": loss, "mmd_weight": weight}

@register_loss("vq")
class VQLoss(LossStrategy):
    def __init__(self):
        super().__init__()

    def compute_loss(self, model_output, labels=None, epoch=None, step=None, **kwargs):
        vq_loss = model_output.get("vq_loss", 0.0)
        vq_loss = torch.mean(torch.nan_to_num(vq_loss, nan=0.0, posinf=0.0, neginf=0.0))
        return {
            "loss": vq_loss,
            "vq_loss": vq_loss,
        }
    
@register_loss("margin_contrastive_loss")
class MarginContrastiveLossStrategy(LossStrategy):
    def __init__(self, margin=1.0, normalize=False):
        """
        margin: float — separation margin between positives and negatives
        normalize: bool — whether to L2-normalize embeddings before computing distances
        """
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def compute_loss(self, model_output, labels=None, step=None, epoch=None, **kwargs):
        q = model_output["q"]  # [B, D]
        k = model_output["k"]  # [B, D]

        if self.normalize:
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)

        B = q.size(0)

        # --- Positive distances (anchor vs. positive) ---
        pos_dist = torch.sum((q - k) ** 2, dim=1)  # [B]

        # --- Negative distances (anchor vs. all other positives) ---
        dist_matrix = torch.cdist(q, k, p=2).pow(2)  # [B, B]
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=q.device)
        neg_dists = dist_matrix[neg_mask].view(B, -1)  # [B, B-1]

        # --- Margin-based contrastive term ---
        neg_loss = F.relu(self.margin - neg_dists)  # [B, B-1]
        neg_loss = torch.mean(neg_loss, dim=1)  # average over negatives

        # --- Final loss per sample ---
        loss_per_sample = pos_dist + neg_loss
        loss = loss_per_sample.mean()

        return {
            "loss": loss,
            "margin_contrastive_loss": loss,
        }

    
@register_loss("composite")
class CompositeLossStrategy(LossStrategy):
    def __init__(self, losses_config, reduction="sum"):
        self.reduction = reduction
        self.losses = []
        self.vq_weight = 1.0
        for loss_cfg in losses_config:
            loss_name = loss_cfg["name"]
            weight = loss_cfg.get("weight", 1.0)
            params = loss_cfg.get("params", {})
            loss_cls = modules.LOSS_REGISTRY[loss_name]
            loss_instance = loss_cls(**params)
            self.losses.append((loss_instance, weight))

    def compute_loss(self, model_output, labels, epoch=None, step=None, **kwargs):
        loss_dict = {}
        weighted_losses = []

        for loss_instance, weight in self.losses:
            loss_res = loss_instance.compute_loss(model_output, labels, epoch=epoch, step=step, **kwargs)                
            loss_res["loss"] = torch.mean(torch.nan_to_num(loss_res["loss"], nan=0.0, posinf=0.0, neginf=0.0))
            weighted_losses.append(loss_res["loss"] * weight)

            # accumulate other loss terms without weighting (e.g. raw kl_loss, clf_loss)
            for k, v in loss_res.items():
                if k == "loss":
                    continue
                loss_dict[k] = loss_dict.get(k, 0) + v

        if self.reduction == "sum":
            total_loss = torch.stack(weighted_losses).sum()
        elif self.reduction == "mean":
            total_loss = torch.stack(weighted_losses).mean()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        loss_dict["loss"] = total_loss
        return loss_dict