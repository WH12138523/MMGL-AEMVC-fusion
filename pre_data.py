from typing import Optional

import numpy as np
import torch


def similarity_rbf(x: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    x = x.float()
    dist = torch.cdist(x, x, p=2) ** 2
    if gamma is None:
        gamma = 1.0 / max(x.shape[1], 1)
    return torch.exp(-gamma * dist)


def similarity_cos(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-12)
    return x @ x.t()


def get_k(x: torch.Tensor, kernel_type: str = "rbf") -> torch.Tensor:
    if kernel_type == "rbf":
        return similarity_rbf(x)
    if kernel_type == "cos":
        return similarity_cos(x)
    raise ValueError(f"Unsupported kernel_type={kernel_type}")


def laplacian_from_features(x: torch.Tensor, kernel_type: str = "rbf") -> torch.Tensor:
    s = get_k(x, kernel_type=kernel_type)
    d = torch.diag(torch.sum(s, dim=1))
    return d - s
