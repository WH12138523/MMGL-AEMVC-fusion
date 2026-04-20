import torch


def graph_reg(latent: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    # FUSION MODIFICATION: graph regularization term tr(Z^T L Z)/N.
    n = max(latent.shape[0], 1)
    return torch.trace(latent.t() @ laplacian @ latent) / n


def _center_kernel(k: torch.Tensor) -> torch.Tensor:
    n = k.shape[0]
    h = torch.eye(n, device=k.device, dtype=k.dtype) - torch.ones((n, n), device=k.device, dtype=k.dtype) / n
    return h @ k @ h


def hsic(kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    # FUSION MODIFICATION: normalized HSIC objective.
    n = kx.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=kx.device, dtype=kx.dtype)
    cx = _center_kernel(kx)
    cy = _center_kernel(ky)
    return torch.trace(cx @ cy) / ((n - 1) ** 2)
