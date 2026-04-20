from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GCNLayer


class VLTransformer(nn.Module):
    # FUSION MODIFICATION: supports concatenating raw modal features with AEMVC kernel features.
    def __init__(self, modal_input_dims: Dict[str, int], hidden_dim: int = 64, use_kernel_concat: bool = True):
        super().__init__()
        self.modal_names = list(modal_input_dims.keys())
        self.use_kernel_concat = use_kernel_concat
        self.modal_proj = nn.ModuleDict({m: nn.Linear(d, hidden_dim) for m, d in modal_input_dims.items()})
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(
        self,
        modal_data_dict: Dict[str, torch.Tensor],
        kernel_repr_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        reps = []
        for m in self.modal_names:
            x = modal_data_dict[m]
            if self.use_kernel_concat:
                if kernel_repr_dict is not None and m in kernel_repr_dict:
                    k = kernel_repr_dict[m]
                    # FUSION MODIFICATION: stable per-sample kernel feature extraction via row mean pooling.
                    kfeat = torch.mean(k, dim=1, keepdim=True)
                else:
                    # FUSION MODIFICATION: preserve dimensional compatibility when fusion kernels are unavailable.
                    kfeat = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
                x = torch.cat([x, kfeat], dim=1)
            reps.append(self.modal_proj[m](x))
        token = torch.stack(reps, dim=1)
        out, _ = self.attn(token, token, token)
        return torch.mean(out, dim=1)


class GraphLearn(nn.Module):
    # FUSION MODIFICATION: optional AEMVC prior adjacency fusion with bounded learnable beta.
    def __init__(self, input_dim: int, use_prior_fusion: bool = False, init_beta: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.use_prior_fusion = use_prior_fusion
        # FUSION MODIFICATION: numerically stable learnable beta initialization in logit space.
        safe_beta = float(min(max(init_beta, 1e-4), 1.0 - 1e-4))
        self.beta_logits = nn.Parameter(torch.logit(torch.tensor(safe_beta))) if use_prior_fusion else None

    def _weighted_cos_adj(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.weight)
        xw = x * w
        xw = xw / (torch.norm(xw, dim=1, keepdim=True) + 1e-12)
        adj = xw @ xw.t()
        adj = torch.clamp(adj, min=0.0)
        return adj

    def forward(self, x: torch.Tensor, prior_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        a_mmgl = self._weighted_cos_adj(x)
        if not self.use_prior_fusion or prior_adj is None:
            return a_mmgl
        beta = torch.sigmoid(self.beta_logits)
        return beta * a_mmgl + (1.0 - beta) * prior_adj


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.g1 = GCNLayer(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.g1(x, adj)
        return self.out(h)


class GAT(nn.Module):
    # FUSION MODIFICATION: simple alias to keep API compatibility when GAT requested.
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.backbone = GCN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, adj)
