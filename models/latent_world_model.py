"""
Latent World Model for dynamics prediction in compressed representation space.

Learns dynamics in the latent space for more efficient planning and generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple, Dict, List

from .world_model import MLP


class LatentWorldModel(nn.Module):
    """
    World model that operates in latent space.

    Architecture:
    - Encoder: obs → z
    - Dynamics: (z_t, a_t) → z_{t+1}
    - Reward: (z_t, a_t) → r_t
    - Decoder: z → obs (optional, for visualization)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module],
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        predict_delta: bool = True,
        separate_reward_head: bool = True,
    ):
        """
        Initialize latent world model.

        Args:
            encoder: Encoder network (obs → latent)
            decoder: Decoder network (latent → obs), optional
            latent_dim: Dimension of latent space
            action_dim: Dimension of action space
            hidden_dims: Hidden dimensions for dynamics network
            activation: Activation function
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
            predict_delta: Whether to predict latent state changes
            separate_reward_head: Use separate network for reward prediction
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.predict_delta = predict_delta
        self.separate_reward_head = separate_reward_head

        # Freeze encoder/decoder if desired (can be unfrozen for fine-tuning)
        self.freeze_encoder_decoder = False

        # Dynamics network: (z_t, a_t) → z_{t+1}
        self.dynamics_net = MLP(
            input_dim=latent_dim + action_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        if separate_reward_head:
            # Separate reward predictor
            self.reward_net = MLP(
                input_dim=latent_dim + action_dim,
                output_dim=1,
                hidden_dims=[hidden_dims[0]],  # Smaller network for reward
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
            )
        else:
            # Joint prediction (dynamics + reward)
            self.joint_net = MLP(
                input_dim=latent_dim + action_dim,
                output_dim=latent_dim + 1,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
            )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to latent space.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            latent: Latent representations [batch_size, latent_dim]
        """
        if self.freeze_encoder_decoder:
            with torch.no_grad():
                return self.encoder(obs)
        return self.encoder(obs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to observations.

        Args:
            latent: Latent representations [batch_size, latent_dim]

        Returns:
            obs: Reconstructed observations [batch_size, obs_dim]
        """
        if self.decoder is None:
            raise ValueError("No decoder available")

        if self.freeze_encoder_decoder:
            with torch.no_grad():
                return self.decoder(latent)
        return self.decoder(latent)

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next latent state and reward.

        Args:
            latent: Current latent state [batch_size, latent_dim]
            action: Action [batch_size, action_dim]

        Returns:
            next_latent: Next latent state [batch_size, latent_dim]
            reward: Predicted reward [batch_size, 1]
        """
        # Concatenate latent and action
        x = torch.cat([latent, action], dim=-1)

        if self.separate_reward_head:
            # Separate predictions
            next_latent_pred = self.dynamics_net(x)
            reward_pred = self.reward_net(x)
        else:
            # Joint prediction
            output = self.joint_net(x)
            next_latent_pred = output[..., :self.latent_dim]
            reward_pred = output[..., self.latent_dim:]

        # Apply delta prediction if configured
        if self.predict_delta:
            next_latent = latent + next_latent_pred
        else:
            next_latent = next_latent_pred

        return next_latent, reward_pred

    def forward_from_obs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass starting from observations.

        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action [batch_size, action_dim]

        Returns:
            next_obs: Predicted next observation [batch_size, obs_dim]
            reward: Predicted reward [batch_size, 1]
        """
        # Encode to latent space
        latent = self.encode(obs)

        # Predict in latent space
        next_latent, reward = self.forward(latent, action)

        # Decode back to observation space
        if self.decoder is not None:
            next_obs = self.decode(next_latent)
        else:
            # If no decoder, return latent as "observation"
            next_obs = next_latent

        return next_obs, reward

    def imagine_trajectory(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
        decode_observations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory in latent space.

        Args:
            initial_latent: Initial latent state [batch_size, latent_dim]
            actions: Action sequence [batch_size, horizon, action_dim]
            decode_observations: Whether to decode latents to observations

        Returns:
            Dictionary containing:
                - latents: [batch_size, horizon+1, latent_dim]
                - rewards: [batch_size, horizon]
                - observations: [batch_size, horizon+1, obs_dim] (if decoded)
        """
        batch_size, horizon, _ = actions.shape

        latents = [initial_latent]
        rewards = []

        latent = initial_latent
        for t in range(horizon):
            action = actions[:, t]
            next_latent, reward = self.forward(latent, action)

            latents.append(next_latent)
            rewards.append(reward.squeeze(-1))

            latent = next_latent

        latents = torch.stack(latents, dim=1)
        rewards = torch.stack(rewards, dim=1)

        result = {
            "latents": latents,
            "rewards": rewards,
        }

        # Optionally decode to observations
        if decode_observations and self.decoder is not None:
            # Reshape for batch decoding
            latents_flat = latents.view(-1, self.latent_dim)
            obs_flat = self.decode(latents_flat)
            obs_dim = obs_flat.shape[-1]
            observations = obs_flat.view(batch_size, horizon + 1, obs_dim)
            result["observations"] = observations

        return result

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        beta_recon: float = 1.0,
        beta_dynamics: float = 1.0,
        beta_reward: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for latent world model training.

        Args:
            obs: Current observations
            action: Actions
            next_obs: True next observations
            reward: True rewards
            beta_recon: Weight for reconstruction loss
            beta_dynamics: Weight for dynamics loss
            beta_reward: Weight for reward loss

        Returns:
            Dictionary of losses
        """
        # Encode current and next observations
        latent = self.encode(obs)
        next_latent_true = self.encode(next_obs)

        # Predict next latent and reward
        next_latent_pred, reward_pred = self.forward(latent, action)

        # Dynamics loss in latent space
        if self.predict_delta:
            delta_true = next_latent_true - latent
            delta_pred = next_latent_pred - latent
            dynamics_loss = F.mse_loss(delta_pred, delta_true)
        else:
            dynamics_loss = F.mse_loss(next_latent_pred, next_latent_true)

        # Reward loss
        reward_loss = F.mse_loss(reward_pred, reward.unsqueeze(-1))

        # Optional reconstruction loss
        recon_loss = torch.tensor(0.0, device=obs.device)
        if self.decoder is not None and beta_recon > 0:
            obs_recon = self.decode(latent)
            next_obs_recon = self.decode(next_latent_pred)

            recon_loss = F.mse_loss(obs_recon, obs) + F.mse_loss(next_obs_recon, next_obs)

        # Total loss
        total_loss = (
            beta_dynamics * dynamics_loss +
            beta_reward * reward_loss +
            beta_recon * recon_loss
        )

        return {
            "loss": total_loss,
            "dynamics_loss": dynamics_loss,
            "reward_loss": reward_loss,
            "recon_loss": recon_loss,
        }

    def set_encoder_decoder_freeze(self, freeze: bool):
        """
        Freeze or unfreeze encoder/decoder weights.

        Args:
            freeze: Whether to freeze encoder/decoder
        """
        self.freeze_encoder_decoder = freeze

        # Also set requires_grad for clarity
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = not freeze


class LatentEnsembleWorldModel(nn.Module):
    """
    Ensemble of latent world models for uncertainty estimation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module],
        latent_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        **kwargs
    ):
        """
        Initialize ensemble latent world model.

        Args:
            encoder: Shared encoder
            decoder: Shared decoder
            latent_dim: Latent dimension
            action_dim: Action dimension
            ensemble_size: Number of models in ensemble
            **kwargs: Arguments for individual models
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Create ensemble of dynamics models (share encoder/decoder)
        self.models = nn.ModuleList([
            LatentWorldModel(
                encoder=encoder,
                decoder=decoder,
                latent_dim=latent_dim,
                action_dim=action_dim,
                **kwargs
            )
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        reduce: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.

        Args:
            latent: Current latent state
            action: Action
            reduce: Reduction method ("none", "mean", "mean_std")

        Returns:
            Next latent and reward predictions
        """
        predictions = []
        for model in self.models:
            next_latent, reward = model(latent, action)
            predictions.append((next_latent, reward))

        # Stack predictions
        next_latent_all = torch.stack([p[0] for p in predictions])
        rewards_all = torch.stack([p[1] for p in predictions])

        if reduce == "none":
            return next_latent_all, rewards_all
        elif reduce == "mean":
            return next_latent_all.mean(dim=0), rewards_all.mean(dim=0)
        elif reduce == "mean_std":
            next_latent_mean = next_latent_all.mean(dim=0)
            next_latent_std = next_latent_all.std(dim=0)
            rewards_mean = rewards_all.mean(dim=0)
            rewards_std = rewards_all.std(dim=0)
            return (next_latent_mean, next_latent_std), (rewards_mean, rewards_std)
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations using shared encoder."""
        return self.encoder(obs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents using shared decoder."""
        if self.decoder is None:
            raise ValueError("No decoder available")
        return self.decoder(latent)


class StochasticLatentWorldModel(LatentWorldModel):
    """
    Latent world model with stochastic dynamics (Gaussian).
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module],
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        predict_delta: bool = True,
        separate_reward_head: bool = True,
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            predict_delta=predict_delta,
            separate_reward_head=separate_reward_head,
        )

        input_dim = latent_dim + action_dim

        # Rebuild dynamics net to output mean and log_std
        # Output dim is 2 * latent_dim (mean + log_std)
        self.dynamics_net = MLP(
            input_dim=input_dim,
            output_dim=2 * latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        
        # Reward net remains deterministic for now, or could be stochastic too
        # Keeping reward deterministic as per plan "minimal Gaussian stochastic WM" usually refers to state

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        """
        Predict next latent state and reward.

        Args:
            latent: Current latent state
            action: Action
            deterministic: If True, return mean

        Returns:
            next_latent: Sampled next latent state
            reward: Predicted reward
            dist: Distribution over next latent state (or delta)
        """
        x = torch.cat([latent, action], dim=-1)

        # Dynamics
        out = self.dynamics_net(x)
        mean = out[..., :self.latent_dim]
        log_std = out[..., self.latent_dim:]
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, min=-10.0, max=2.0)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        
        if deterministic:
            sample = mean
        else:
            sample = dist.rsample()
            
        if self.predict_delta:
            next_latent = latent + sample
        else:
            next_latent = sample
            
        # Reward
        if self.separate_reward_head:
            reward = self.reward_net(x)
        else:
            # If we were using joint net, we'd need to change it too. 
            # Assuming separate_reward_head=True for stochastic model for simplicity
            raise NotImplementedError("Stochastic model requires separate_reward_head=True")
            
        return next_latent, reward, dist

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        beta_recon: float = 1.0,
        beta_dynamics: float = 1.0,
        beta_reward: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NLL loss for stochastic model.
        """
        latent = self.encode(obs)
        next_latent_true = self.encode(next_obs)
        
        # Forward (deterministic=False doesn't matter for distribution retrieval)
        next_latent_pred, reward_pred, dist = self.forward(latent, action, deterministic=False)
        
        # NLL Loss
        if self.predict_delta:
            target = next_latent_true - latent
        else:
            target = next_latent_true
            
        # Negative Log Likelihood
        nll = -dist.log_prob(target).sum(dim=-1).mean()
        
        # Reward loss (MSE)
        reward_loss = F.mse_loss(reward_pred, reward.unsqueeze(-1))
        
        # Reconstruction loss
        recon_loss = torch.tensor(0.0, device=obs.device)
        if self.decoder is not None and beta_recon > 0:
            # Use mean for reconstruction
            if self.predict_delta:
                next_latent_mean = latent + dist.mean
            else:
                next_latent_mean = dist.mean
                
            obs_recon = self.decode(latent)
            next_obs_recon = self.decode(next_latent_mean)
            recon_loss = F.mse_loss(obs_recon, obs) + F.mse_loss(next_obs_recon, next_obs)
            
        total_loss = (
            beta_dynamics * nll +
            beta_reward * reward_loss +
            beta_recon * recon_loss
        )
        
        return {
            "loss": total_loss,
            "nll_loss": nll,
            "reward_loss": reward_loss,
            "recon_loss": recon_loss,
        }