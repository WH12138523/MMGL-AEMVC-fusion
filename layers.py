import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    # FUSION MODIFICATION: lightweight GCN layer fallback (DGL-independent).
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        deg = torch.sum(adj, dim=1) + 1e-12
        deg_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt
        return F.relu(norm_adj @ h)
