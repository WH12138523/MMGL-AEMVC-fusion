import torch
import torch.nn as nn


class Net(nn.Module):
    # FUSION MODIFICATION: reusable modality-specific autoencoder for AEMVC.
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        hidden = max(32, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon
