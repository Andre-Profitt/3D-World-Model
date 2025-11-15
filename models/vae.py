"""
Variational Autoencoder for improved latent representation learning.

Key improvements over basic autoencoder:
- Stochastic latent space with KL regularization
- Better capacity and architecture
- Learnable variance for robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VAEEncoder(nn.Module):
    """
    Variational encoder that outputs distribution parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list = [256, 256, 256],
        activation: str = "elu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Build encoder network
        dims = [obs_dim] + hidden_dims
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            if activation == "elu":
                layers.append(nn.ELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

        self.encoder = nn.Sequential(*layers)

        # Output heads for mean and log variance
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent distribution parameters.

        Args:
            x: Observation tensor [batch_size, obs_dim]

        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp log variance for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Variational decoder that reconstructs observations from latent codes.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dims: list = [256, 256, 256],
        activation: str = "elu",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Build decoder network
        dims = [latent_dim] + hidden_dims + [obs_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/norm on output layer
                layers.append(nn.LayerNorm(dims[i + 1]))
                if activation == "elu":
                    layers.append(nn.ELU())
                elif activation == "relu":
                    layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to observation.

        Args:
            z: Latent code [batch_size, latent_dim]

        Returns:
            Reconstructed observation [batch_size, obs_dim]
        """
        return self.decoder(z)


class VAE(nn.Module):
    """
    Complete Variational Autoencoder.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        encoder_hidden: list = [256, 256, 256],
        decoder_hidden: list = [256, 256, 256],
        activation: str = "elu",
        beta: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.beta = beta  # KL weighting factor (beta-VAE)

        # Create encoder and decoder
        self.encoder = VAEEncoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden,
            activation=activation,
        )

        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            hidden_dims=decoder_hidden,
            activation=activation,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of distribution
            logvar: Log variance of distribution

        Returns:
            Sampled latent code
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Use mean during evaluation for deterministic behavior
            return mu

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input observation

        Returns:
            recon: Reconstructed observation
            mu: Latent distribution mean
            logvar: Latent distribution log variance
            z: Sampled latent code
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decoder(z)

        return recon, mu, logvar, z

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """
        Encode observation to latent code.

        Args:
            x: Observation
            sample: Whether to sample or return mean

        Returns:
            Latent code
        """
        mu, logvar = self.encoder(x)
        if sample and self.training:
            return self.reparameterize(mu, logvar)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to observation.

        Args:
            z: Latent code

        Returns:
            Reconstructed observation
        """
        return self.decoder(z)

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original observation
            recon: Reconstructed observation
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            Total loss and component dictionary
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss

        # Return components for logging
        components = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
        }

        return total_loss, components


class ConvVAEEncoder(nn.Module):
    """
    Convolutional VAE encoder for image observations.
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        latent_dim: int = 32,
    ):
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.ReLU(),
        )

        # Calculate flattened dimension
        self.flatten_dim = 256 * (img_size // 16) * (img_size // 16)

        # FC layers for distribution parameters
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.

        Args:
            x: Image tensor [batch, channels, height, width]

        Returns:
            mu, logvar of latent distribution
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar


class ConvVAEDecoder(nn.Module):
    """
    Convolutional VAE decoder for image generation.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        img_channels: int = 3,
        img_size: int = 64,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Initial size after reshape
        self.init_size = img_size // 16  # 4 for 64x64 images
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size * self.init_size),
            nn.ReLU(),
        )

        # Deconvolutional layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to image.

        Args:
            z: Latent code [batch, latent_dim]

        Returns:
            Reconstructed image [batch, channels, height, width]
        """
        h = self.fc(z)
        h = h.view(h.size(0), 256, self.init_size, self.init_size)
        return self.deconv(h)