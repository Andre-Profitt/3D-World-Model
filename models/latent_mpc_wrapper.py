"""
Wrapper for latent world model to work with MPC controller.

Provides a unified interface for planning in latent space while maintaining
compatibility with the existing MPC implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LatentMPCWrapper(nn.Module):
    """
    Wrapper that combines encoder, latent world model, and decoder
    for use with MPC planning.

    This implements the full V-M-C (Vision-Model-Controller) architecture:
    - V (Vision/Encoder): Encodes observations to latent space
    - M (Model): Predicts dynamics in latent space
    - C (Controller): MPC plans in latent space
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_world_model: nn.Module,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize latent MPC wrapper.

        Args:
            encoder: Trained encoder (obs -> latent)
            decoder: Trained decoder (latent -> obs)
            latent_world_model: Trained dynamics model in latent space
            device: Device to run on
        """
        super().__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.latent_model = latent_world_model.to(device)
        self.device = device

        # Set to eval mode by default
        self.encoder.eval()
        self.decoder.eval()
        self.latent_model.eval()

        # Cache dimensions
        self.obs_dim = encoder.obs_dim if hasattr(encoder, 'obs_dim') else decoder.obs_dim
        self.latent_dim = encoder.latent_dim if hasattr(encoder, 'latent_dim') else decoder.latent_dim
        self.action_dim = latent_world_model.action_dim

        # Current latent state (for stateful planning)
        self.current_latent = None

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space."""
        with torch.no_grad():
            latent = self.encoder(obs)
            # Handle both regular encoder and VAE encoder outputs
            if isinstance(latent, tuple):
                latent = latent[0]  # Use mean for VAE
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation space."""
        with torch.no_grad():
            obs = self.decoder(latent)
            # Handle both regular decoder and VAE decoder outputs
            if isinstance(obs, tuple):
                obs = obs[0]  # Use mean for VAE
        return obs

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        decode_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the latent world model.

        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action to take [batch_size, action_dim]
            decode_output: Whether to decode the predicted latent back to obs space

        Returns:
            next_obs: Predicted next observation (or latent if decode_output=False)
            reward: Predicted reward
        """
        # Ensure inputs are on the correct device
        obs = obs.to(self.device)
        action = action.to(self.device)

        # Encode current observation to latent
        latent = self.encode(obs)

        # Predict next latent and reward
        next_latent, reward = self.latent_model(latent, action)

        # Optionally decode back to observation space
        if decode_output:
            next_obs = self.decode(next_latent)
        else:
            next_obs = next_latent

        return next_obs, reward

    def reset_latent(self, obs: Optional[torch.Tensor] = None):
        """
        Reset the current latent state.

        Args:
            obs: Initial observation to encode (optional)
        """
        if obs is not None:
            self.current_latent = self.encode(obs)
        else:
            self.current_latent = None

    def step_latent(
        self,
        action: torch.Tensor,
        return_obs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step forward using cached latent state (more efficient for sequential planning).

        Args:
            action: Action to take
            return_obs: Whether to return decoded observation or raw latent

        Returns:
            next_state: Next observation or latent
            reward: Predicted reward
        """
        if self.current_latent is None:
            raise ValueError("Must call reset_latent() with initial observation first")

        action = action.to(self.device)

        # Predict in latent space
        next_latent, reward = self.latent_model(self.current_latent, action)

        # Update cached latent
        self.current_latent = next_latent

        # Return decoded or latent
        if return_obs:
            next_obs = self.decode(next_latent)
            return next_obs, reward
        else:
            return next_latent, reward

    def imagine_trajectory(
        self,
        initial_obs: torch.Tensor,
        actions: torch.Tensor,
        decode_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Imagine a trajectory by rolling out actions from initial observation.

        Args:
            initial_obs: Initial observation [batch_size, obs_dim]
            actions: Action sequence [batch_size, horizon, action_dim]
            decode_trajectory: Whether to decode the trajectory to observation space

        Returns:
            trajectory: Predicted states [batch_size, horizon+1, state_dim]
            rewards: Predicted rewards [batch_size, horizon]
        """
        batch_size, horizon, _ = actions.shape

        # Encode initial observation
        latent = self.encode(initial_obs.to(self.device))

        latent_trajectory = [latent]
        rewards = []

        # Roll out in latent space
        for t in range(horizon):
            action = actions[:, t].to(self.device)
            next_latent, reward = self.latent_model(latent, action)

            latent_trajectory.append(next_latent)
            rewards.append(reward.squeeze(-1))

            latent = next_latent

        # Stack trajectory
        latent_trajectory = torch.stack(latent_trajectory, dim=1)
        rewards = torch.stack(rewards, dim=1)

        # Optionally decode trajectory
        if decode_trajectory:
            obs_trajectory = []
            for t in range(latent_trajectory.shape[1]):
                obs = self.decode(latent_trajectory[:, t])
                obs_trajectory.append(obs)
            trajectory = torch.stack(obs_trajectory, dim=1)
        else:
            trajectory = latent_trajectory

        return trajectory, rewards