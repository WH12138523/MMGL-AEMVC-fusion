import torch.nn as nn

from aemvc_fusion import AEMVCFusion


class AEMVC(nn.Module):
    # FUSION MODIFICATION: compatibility class preserving legacy AEMVC entrypoint.
    def __init__(self, modal_dims, latent_dim=64, kernel_type="rbf", lambda1=1.0, lambda2=1.0, lambda3=1.0):
        super().__init__()
        self.model = AEMVCFusion(
            modal_dims=modal_dims,
            latent_dim=latent_dim,
            kernel_type=kernel_type,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            use_kcca=False,
        )

    def forward(self, modal_data_dict, missing_mask_dict):
        return self.model(modal_data_dict, missing_mask_dict)
