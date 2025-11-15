"""
Stochastic Variational Autoencoder World Model.

Combines VAE with stochastic dynamics for improved representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Tuple, Dict, Optional, List
import numpy as np


class StochasticEncoder(nn.Module):
    """Stochastic encoder that outputs distribution parameters."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        layer_norm: bool = False,
        dropout: float = 0.0,
        min_std: float = 0.01,
    ):
        """
        Initialize stochastic encoder.

        Args:
            input_dim: Input observation dimension
            latent_dim: Latent representation dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            layer_norm: Whether to use layer normalization
            dropout: Dropout probability
            min_std: Minimum standard deviation
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.min_std = min_std

        # Build encoder network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.encoder_net = nn.Sequential(*layers)

        # Output heads for mean and log variance
        self.mean_head = nn.Linear(prev_dim, latent_dim)
        self.logvar_head = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent distribution parameters.

        Args:
            x: Input observation

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder_net(x)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)

        # Bound log variance for stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar

    def sample(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            mu: Mean of distribution
            logvar: Log variance of distribution
            deterministic: If True, return mean

        Returns:
            Sampled latent vector
        """
        if deterministic:
            return mu

        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=self.min_std)
        eps = torch.randn_like(std)
        return mu + eps * std


class StochasticDecoder(nn.Module):
    """Stochastic decoder that reconstructs observations from latents."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        layer_norm: bool = False,
        dropout: float = 0.0,
        output_std: Optional[float] = None,
    ):
        """
        Initialize stochastic decoder.

        Args:
            latent_dim: Latent representation dimension
            output_dim: Output observation dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            layer_norm: Whether to use layer normalization
            dropout: Dropout probability
            output_std: Fixed output standard deviation (None for learned)
        """
        super().__init__()

        self.output_dim = output_dim
        self.output_std = output_std

        # Build decoder network
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.decoder_net = nn.Sequential(*layers)

        # Output mean
        self.mean_head = nn.Linear(prev_dim, output_dim)

        # Output log variance (if not fixed)
        if output_std is None:
            self.logvar_head = nn.Linear(prev_dim, output_dim)
        else:
            self.logvar_head = None

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to observation distribution parameters.

        Args:
            z: Latent vector

        Returns:
            mu: Mean of output distribution
            logvar: Log variance of output distribution
        """
        h = self.decoder_net(z)
        mu = self.mean_head(h)

        if self.logvar_head is not None:
            logvar = self.logvar_head(h)
            logvar = torch.clamp(logvar, min=-10, max=10)
        else:
            # Fixed variance
            logvar = torch.ones_like(mu) * np.log(self.output_std ** 2)

        return mu, logvar


class StochasticVAEDynamics(nn.Module):
    """
    Stochastic dynamics model in VAE latent space.

    Predicts next latent and reward distributions.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        predict_delta: bool = True,
        separate_reward_head: bool = True,
        min_std: float = 0.01,
        max_std: float = 1.0,
    ):
        """
        Initialize stochastic VAE dynamics.

        Args:
            latent_dim: Latent representation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            predict_delta: Whether to predict latent changes
            separate_reward_head: Use separate network for rewards
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.predict_delta = predict_delta
        self.separate_reward_head = separate_reward_head
        self.min_std = min_std
        self.max_std = max_std

        input_dim = latent_dim + action_dim

        # Build shared layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        layers.append(nn.ReLU())

        self.shared_net = nn.Sequential(*layers)

        # Latent dynamics heads
        self.latent_mean_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.latent_logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

        if separate_reward_head:
            # Separate reward network
            self.reward_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
                nn.ReLU(),
            )
            self.reward_mean_head = nn.Linear(hidden_dims[0] // 2, 1)
            self.reward_logvar_head = nn.Linear(hidden_dims[0] // 2, 1)
        else:
            # Joint reward heads
            self.reward_mean_head = nn.Linear(hidden_dims[-1], 1)
            self.reward_logvar_head = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict next latent and reward distributions.

        Args:
            z: Current latent
            action: Action
            deterministic: Use mean predictions

        Returns:
            next_z: Predicted next latent
            reward: Predicted reward
            dist_params: Distribution parameters
        """
        # Concatenate inputs
        x = torch.cat([z, action], dim=-1)

        # Latent dynamics
        h = self.shared_net(x)
        latent_mean = self.latent_mean_head(h)
        latent_logvar = self.latent_logvar_head(h)
        latent_logvar = torch.clamp(latent_logvar, min=-10, max=10)

        # Apply delta prediction
        if self.predict_delta:
            latent_mean = z + latent_mean

        # Reward prediction
        if self.separate_reward_head:
            h_reward = self.reward_net(x)
            reward_mean = self.reward_mean_head(h_reward)
            reward_logvar = self.reward_logvar_head(h_reward)
        else:
            reward_mean = self.reward_mean_head(h)
            reward_logvar = self.reward_logvar_head(h)

        reward_logvar = torch.clamp(reward_logvar, min=-10, max=10)

        # Sample or use mean
        if deterministic:
            next_z = latent_mean
            reward = reward_mean
        else:
            # Sample from distributions
            latent_std = torch.exp(0.5 * latent_logvar)
            latent_std = torch.clamp(latent_std, self.min_std, self.max_std)
            latent_dist = Normal(latent_mean, latent_std)
            next_z = latent_dist.rsample()

            reward_std = torch.exp(0.5 * reward_logvar)
            reward_std = torch.clamp(reward_std, self.min_std, self.max_std)
            reward_dist = Normal(reward_mean, reward_std)
            reward = reward_dist.rsample()

        dist_params = {
            "latent_mean": latent_mean,
            "latent_logvar": latent_logvar,
            "reward_mean": reward_mean,
            "reward_logvar": reward_logvar,
        }

        return next_z, reward, dist_params


class StochasticVAEWorldModel(nn.Module):
    """
    Complete Stochastic VAE World Model.

    Combines encoder, decoder, and dynamics for full world modeling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        encoder_hidden: List[int] = [256, 256],
        decoder_hidden: List[int] = [256, 256],
        dynamics_hidden: List[int] = [256, 256],
        activation: str = "relu",
        predict_delta: bool = True,
        beta: float = 1.0,
        free_nats: float = 3.0,
        reconstruction_loss: str = "mse",
    ):
        """
        Initialize Stochastic VAE World Model.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            latent_dim: Latent representation dimension
            encoder_hidden: Encoder hidden dimensions
            decoder_hidden: Decoder hidden dimensions
            dynamics_hidden: Dynamics hidden dimensions
            activation: Activation function
            predict_delta: Predict latent changes
            beta: KL divergence weight
            free_nats: Free nats for KL (minimum KL value)
            reconstruction_loss: Type of reconstruction loss
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.free_nats = free_nats
        self.reconstruction_loss = reconstruction_loss

        # Encoder: obs -> latent distribution
        self.encoder = StochasticEncoder(
            input_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden,
            activation=activation,
        )

        # Decoder: latent -> obs distribution
        self.decoder = StochasticDecoder(
            latent_dim=latent_dim,
            output_dim=obs_dim,
            hidden_dims=decoder_hidden,
            activation=activation,
            output_std=0.1 if reconstruction_loss == "gaussian_nll" else None,
        )

        # Dynamics: (latent, action) -> next latent, reward
        self.dynamics = StochasticVAEDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=dynamics_hidden,
            activation=activation,
            predict_delta=predict_delta,
        )

    def encode(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent.

        Args:
            obs: Observation
            deterministic: Use mean encoding

        Returns:
            z: Latent sample
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encoder(obs)
        z = self.encoder.sample(mu, logvar, deterministic)
        return z, mu, logvar

    def decode(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to observation.

        Args:
            z: Latent vector

        Returns:
            obs_mu: Observation mean
            obs_logvar: Observation log variance
        """
        return self.decoder(z)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass through VAE world model.

        Args:
            obs: Current observation
            action: Action
            deterministic: Use deterministic predictions

        Returns:
            next_obs: Predicted next observation
            reward: Predicted reward
            info: Dictionary with latents and distributions
        """
        # Encode current observation
        z, z_mu, z_logvar = self.encode(obs, deterministic)

        # Predict next latent and reward
        next_z, reward, dynamics_params = self.dynamics(
            z, action, deterministic
        )

        # Decode next observation
        next_obs_mu, next_obs_logvar = self.decode(next_z)

        if deterministic:
            next_obs = next_obs_mu
        else:
            next_obs_std = torch.exp(0.5 * next_obs_logvar)
            next_obs_dist = Normal(next_obs_mu, next_obs_std)
            next_obs = next_obs_dist.rsample()

        info = {
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "next_z": next_z,
            "next_obs_mu": next_obs_mu,
            "next_obs_logvar": next_obs_logvar,
            **dynamics_params,
        }

        return next_obs, reward, info

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE world model loss.

        Args:
            obs: Current observations
            action: Actions
            next_obs: Next observations
            reward: Rewards

        Returns:
            Dictionary of loss components
        """
        batch_size = obs.shape[0]

        # Encode current and next observations
        z, z_mu, z_logvar = self.encode(obs)
        next_z_target, next_z_mu, next_z_logvar = self.encode(next_obs)

        # KL divergence for encoder (with free nats)
        kl_div = -0.5 * torch.sum(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1
        )
        kl_div = torch.clamp(kl_div, min=self.free_nats)
        kl_loss = kl_div.mean()

        # Reconstruction loss for current observation
        obs_recon_mu, obs_recon_logvar = self.decode(z)

        if self.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(obs_recon_mu, obs, reduction="mean")
        elif self.reconstruction_loss == "gaussian_nll":
            obs_recon_dist = Normal(
                obs_recon_mu,
                torch.exp(0.5 * obs_recon_logvar)
            )
            recon_loss = -obs_recon_dist.log_prob(obs).mean()

        # Dynamics prediction
        next_z_pred, reward_pred, dynamics_params = self.dynamics(
            z, action
        )

        # Latent dynamics loss
        latent_mean = dynamics_params["latent_mean"]
        latent_logvar = dynamics_params["latent_logvar"]
        latent_dist = Normal(
            latent_mean,
            torch.exp(0.5 * latent_logvar)
        )
        latent_dynamics_loss = -latent_dist.log_prob(next_z_target).mean()

        # Reward prediction loss
        reward_mean = dynamics_params["reward_mean"]
        reward_logvar = dynamics_params["reward_logvar"]
        reward_dist = Normal(
            reward_mean,
            torch.exp(0.5 * reward_logvar)
        )
        reward_loss = -reward_dist.log_prob(reward.unsqueeze(-1)).mean()

        # Next observation reconstruction loss
        next_obs_recon_mu, next_obs_recon_logvar = self.decode(next_z_pred)

        if self.reconstruction_loss == "mse":
            next_recon_loss = F.mse_loss(
                next_obs_recon_mu, next_obs, reduction="mean"
            )
        elif self.reconstruction_loss == "gaussian_nll":
            next_obs_recon_dist = Normal(
                next_obs_recon_mu,
                torch.exp(0.5 * next_obs_recon_logvar)
            )
            next_recon_loss = -next_obs_recon_dist.log_prob(next_obs).mean()

        # Total loss
        total_loss = (
            recon_loss +
            next_recon_loss +
            self.beta * kl_loss +
            latent_dynamics_loss +
            reward_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "next_recon_loss": next_recon_loss,
            "kl_loss": kl_loss,
            "latent_dynamics_loss": latent_dynamics_loss,
            "reward_loss": reward_loss,
        }

    def imagine_trajectory(
        self,
        initial_obs: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine trajectory using learned model.

        Args:
            initial_obs: Initial observation
            actions: Action sequence [batch_size, horizon, action_dim]
            deterministic: Use deterministic predictions

        Returns:
            Imagined trajectory information
        """
        batch_size, horizon, _ = actions.shape

        # Encode initial observation
        z, _, _ = self.encode(initial_obs, deterministic)

        observations = [initial_obs]
        latents = [z]
        rewards = []

        for t in range(horizon):
            action = actions[:, t]

            # Predict next latent and reward
            next_z, reward, _ = self.dynamics(z, action, deterministic)

            # Decode to observation
            next_obs_mu, next_obs_logvar = self.decode(next_z)

            if deterministic:
                next_obs = next_obs_mu
            else:
                next_obs_std = torch.exp(0.5 * next_obs_logvar)
                next_obs_dist = Normal(next_obs_mu, next_obs_std)
                next_obs = next_obs_dist.sample()

            observations.append(next_obs)
            latents.append(next_z)
            rewards.append(reward.squeeze(-1))

            z = next_z

        return {
            "observations": torch.stack(observations, dim=1),
            "latents": torch.stack(latents, dim=1),
            "rewards": torch.stack(rewards, dim=1),
        }