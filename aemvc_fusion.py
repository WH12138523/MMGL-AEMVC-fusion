from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from NetModel import Net
from complete import complete
from loss_fun import graph_reg, hsic
from pre_data import get_k, laplacian_from_features


class AEMVCFusion(nn.Module):
    # FUSION MODIFICATION: full AEMVC module reusable in standalone and joint MMGL training.
    def __init__(
        self,
        modal_dims: Dict[str, int],
        latent_dim: int = 64,
        kernel_type: str = "rbf",
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        use_kcca: bool = False,
        jitter: float = 1e-5,
    ):
        super().__init__()
        self.modal_dims = dict(modal_dims)
        self.kernel_type = kernel_type
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.use_kcca = bool(use_kcca)
        self.jitter = float(jitter)
        self.autoencoders = nn.ModuleDict({m: Net(d, latent_dim=latent_dim) for m, d in self.modal_dims.items()})

    def _get_observed_feature_mask(self, modal: str, modal_x: torch.Tensor, missing_mask_dict: Dict) -> torch.Tensor:
        # FUSION MODIFICATION: supports structured dict from enhanced missing generation and legacy masks.
        feat_mask = None
        if missing_mask_dict is not None:
            mfo = missing_mask_dict.get("modal_feature_observed")
            if isinstance(mfo, dict) and modal in mfo:
                feat_mask = mfo[modal]
        if feat_mask is None:
            feat_mask = ~torch.isnan(modal_x)
        if not torch.is_tensor(feat_mask):
            feat_mask = torch.tensor(feat_mask, device=modal_x.device, dtype=torch.bool)
        return feat_mask.bool()

    def _kernel_to_feature_reconstruction(self, kernel: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        # FUSION MODIFICATION: stable practical inverse mapping from completed kernel to feature space.
        reg = self.jitter * torch.eye(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
        try:
            coef = torch.linalg.solve(kernel + reg, x_ref)
        except Exception:
            coef = torch.linalg.pinv(kernel + reg) @ x_ref
        x_hat = kernel @ coef
        return torch.nan_to_num(x_hat)

    def _kcca_hook(self, kernels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # FUSION MODIFICATION: optional KCCA-style placeholder; defaults to centered kernels if enabled.
        if not self.use_kcca:
            return kernels
        out = {}
        for m, k in kernels.items():
            n = k.shape[0]
            h = torch.eye(n, device=k.device, dtype=k.dtype) - torch.ones((n, n), device=k.device, dtype=k.dtype) / n
            out[m] = h @ k @ h
        return out

    def forward(
        self,
        modal_data_dict: Dict[str, torch.Tensor],
        missing_mask_dict: Optional[Dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        recon_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        g_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        hsic_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        kcomp_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        imputed_modal_data: Dict[str, torch.Tensor] = {}
        kernels: Dict[str, torch.Tensor] = {}
        latents: Dict[str, torch.Tensor] = {}

        for modal, x in modal_data_dict.items():
            x = x.float()
            feat_mask = self._get_observed_feature_mask(modal, x, missing_mask_dict or {})
            x_filled = torch.where(feat_mask, x, torch.zeros_like(x))
            z, x_recon = self.autoencoders[modal](x_filled)
            latents[modal] = z
            if feat_mask.any():
                recon_loss = recon_loss + F.mse_loss(x_recon[feat_mask], x[feat_mask])
            lap = laplacian_from_features(x_filled, kernel_type=self.kernel_type)
            g_loss = g_loss + graph_reg(z, lap)
            k = get_k(z, kernel_type=self.kernel_type)
            sample_obs = feat_mask.all(dim=1)
            k_comp = complete(k, observed_sample_mask=sample_obs, laplacian=lap, jitter=self.jitter)
            kernels[modal] = k_comp
            kcomp_loss = kcomp_loss + F.mse_loss(k_comp, k)
            x_kernel_rec = self._kernel_to_feature_reconstruction(k_comp, x_filled)
            imputed_modal_data[modal] = torch.where(feat_mask, x, x_kernel_rec)

        mods = list(kernels.keys())
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                hsic_loss = hsic_loss + (-hsic(kernels[mods[i]], kernels[mods[j]]))

        kernels = self._kcca_hook(kernels)
        total_loss = recon_loss + self.lambda1 * g_loss + self.lambda2 * hsic_loss + self.lambda3 * kcomp_loss
        if torch.isnan(total_loss):
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1.0, neginf=-1.0)
        return imputed_modal_data, kernels, total_loss
