import torch

from complete import complete


def kcomplete(kernel: torch.Tensor, observed_sample_mask: torch.Tensor, laplacian=None) -> torch.Tensor:
    # FUSION MODIFICATION: compatibility wrapper around robust completion.
    return complete(kernel, observed_sample_mask=observed_sample_mask, laplacian=laplacian)
