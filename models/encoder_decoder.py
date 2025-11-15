"""
Encoder and Decoder modules for latent representation learning.

Maps observations to/from a compressed latent space for efficient dynamics modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class Encoder(nn.Module):
    """
    Encoder network that maps observations to latent representations.

    Compresses high-dimensional observations into compact latent codes
    that capture the essential state information.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize encoder.

        Args:
            obs_dim: Dimension of observations
            latent_dim: Dimension of latent representation
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("relu", "elu", "tanh")
            layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to latent representations.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            latent: Latent representations [batch_size, latent_dim]
        """
        return self.encoder(obs)


class Decoder(nn.Module):
    """
    Decoder network that reconstructs observations from latent representations.

    Maps compact latent codes back to the original observation space.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize decoder.

        Args:
            latent_dim: Dimension of latent representation
            obs_dim: Dimension of observations
            hidden_dims: Hidden layer dimensions (reversed from encoder)
            activation: Activation function
            layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Build decoder layers (mirror of encoder)
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer (no activation for reconstruction)
        layers.append(nn.Linear(prev_dim, obs_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to observations.

        Args:
            latent: Latent representations [batch_size, latent_dim]

        Returns:
            obs: Reconstructed observations [batch_size, obs_dim]
        """
        return self.decoder(latent)


class Autoencoder(nn.Module):
    """
    Complete autoencoder combining encoder and decoder.

    Learns compressed representations by reconstruction.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        encoder_hidden_dims: List[int] = [128, 128],
        decoder_hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize autoencoder.

        Args:
            obs_dim: Dimension of observations
            latent_dim: Dimension of latent representation
            encoder_hidden_dims: Encoder hidden layer dimensions
            decoder_hidden_dims: Decoder hidden dims (reverse of encoder if None)
            activation: Activation function
            layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()

        # Default decoder architecture (mirror of encoder)
        if decoder_hidden_dims is None:
            decoder_hidden_dims = list(reversed(encoder_hidden_dims))

        # Create encoder and decoder
        self.encoder = Encoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
        )

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            reconstruction: Reconstructed observations [batch_size, obs_dim]
            latent: Latent representations [batch_size, latent_dim]
        """
        latent = self.encoder(obs)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space."""
        return self.encoder(obs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to observations."""
        return self.decoder(latent)

    def loss(
        self,
        obs: torch.Tensor,
        beta: float = 1.0,
    ) -> dict:
        """
        Compute autoencoder loss.

        Args:
            obs: Observations to reconstruct
            beta: Weight for reconstruction loss

        Returns:
            Dictionary of losses
        """
        # Forward pass
        reconstruction, latent = self.forward(obs)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, obs)

        # Optional: L2 regularization on latent codes
        latent_reg = 0.01 * (latent ** 2).mean()

        # Total loss
        total_loss = beta * recon_loss + latent_reg

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "latent_reg": latent_reg,
        }


class VariationalEncoder(Encoder):
    """
    Variational encoder that outputs distribution parameters.

    For future VAE extension - outputs mean and log variance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace final layer with mean and log_var outputs
        in_features = self.encoder[-1].in_features
        self.encoder = self.encoder[:-1]  # Remove last layer

        self.fc_mu = nn.Linear(in_features, self.latent_dim)
        self.fc_log_var = nn.Linear(in_features, self.latent_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to distribution parameters.

        Args:
            obs: Observations

        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        h = self.encoder(obs)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample from the latent distribution.

        Args:
            obs: Observations
            deterministic: If True, return mean (no sampling)

        Returns:
            Latent sample
        """
        mu, log_var = self.forward(obs)

        if deterministic:
            return mu

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


def create_encoder_decoder_pair(
    obs_dim: int,
    latent_dim: int = 32,
    architecture: str = "shallow",
) -> Tuple[Encoder, Decoder]:
    """
    Create matched encoder-decoder pair with preset architectures.

    Args:
        obs_dim: Observation dimension
        latent_dim: Latent dimension
        architecture: Architecture preset ("shallow", "deep", "wide")

    Returns:
        encoder, decoder tuple
    """
    architectures = {
        "shallow": {
            "encoder_hidden": [64],
            "decoder_hidden": [64],
        },
        "standard": {
            "encoder_hidden": [128, 128],
            "decoder_hidden": [128, 128],
        },
        "deep": {
            "encoder_hidden": [256, 128, 64],
            "decoder_hidden": [64, 128, 256],
        },
        "wide": {
            "encoder_hidden": [512, 256],
            "decoder_hidden": [256, 512],
        },
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}")

    arch = architectures[architecture]

    encoder = Encoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=arch["encoder_hidden"],
    )

    decoder = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dims=arch["decoder_hidden"],
    )

    return encoder, decoder