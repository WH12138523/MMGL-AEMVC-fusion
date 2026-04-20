from typing import Optional

import torch


def complete(
    kernel: torch.Tensor,
    observed_sample_mask: torch.Tensor,
    laplacian: Optional[torch.Tensor] = None,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """
    Laplacian-regularized kernel completion.

    observed_sample_mask: bool[N], True when sample for this modality is observed.
    """
    # FUSION MODIFICATION: robust block-style completion with pseudo-inverse fallback.
    k = kernel.clone()
    n = k.shape[0]
    obs = observed_sample_mask.bool()
    miss = ~obs
    if miss.sum() == 0:
        return k
    if obs.sum() == 0:
        avg = torch.mean(torch.diag(k))
        avg = torch.nan_to_num(avg, nan=1.0, posinf=1.0, neginf=0.0)
        return torch.eye(n, device=k.device, dtype=k.dtype) * avg

    obs_idx = torch.where(obs)[0]
    miss_idx = torch.where(miss)[0]
    koo = k[obs_idx][:, obs_idx]
    kom = k[obs_idx][:, miss_idx]
    kmo = k[miss_idx][:, obs_idx]

    reg = torch.eye(koo.shape[0], device=k.device, dtype=k.dtype) * jitter
    if laplacian is not None:
        loo = laplacian[obs_idx][:, obs_idx]
        reg = reg + jitter * loo

    try:
        inv = torch.linalg.inv(koo + reg)
    except Exception:
        inv = torch.linalg.pinv(koo + reg)

    kmm_hat = kmo @ inv @ kom
    k[miss_idx[:, None], miss_idx[None, :]] = kmm_hat
    k = 0.5 * (k + k.t())
    if torch.isnan(k).any():
        k = torch.nan_to_num(k, nan=0.0, posinf=1.0, neginf=-1.0)
    return k
