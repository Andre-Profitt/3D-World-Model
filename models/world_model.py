"""
World Model for 3D environment dynamics and reward prediction.

Learns to predict next state and reward given current state and action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "tanh", "elu")
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
        """
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class WorldModel(nn.Module):
    """
    World model that predicts next observation and reward.

    Takes (obs_t, action_t) and predicts (obs_{t+1}, reward_t).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        predict_delta: bool = True,
    ):
        """
        Initialize world model.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
            predict_delta: Whether to predict state delta instead of next state
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.predict_delta = predict_delta

        # Input: concatenated observation and action
        input_dim = obs_dim + action_dim

        # Output: next observation + reward
        output_dim = obs_dim + 1

        # Build network
        self.network = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        # Optional: separate heads for state and reward
        self.use_separate_heads = False
        if self.use_separate_heads:
            # Shared trunk
            self.trunk = MLP(
                input_dim=input_dim,
                output_dim=hidden_dims[-1],
                hidden_dims=hidden_dims[:-1],
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            # State prediction head
            self.state_head = nn.Linear(hidden_dims[-1], obs_dim)
            # Reward prediction head
            self.reward_head = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action [batch_size, action_dim]

        Returns:
            next_obs: Predicted next observation [batch_size, obs_dim]
            reward: Predicted reward [batch_size, 1]
        """
        # Concatenate inputs
        x = torch.cat([obs, action], dim=-1)

        if self.use_separate_heads:
            # Shared trunk + separate heads
            features = self.trunk(x)
            next_obs_pred = self.state_head(features)
            reward_pred = self.reward_head(features)
        else:
            # Single network
            output = self.network(x)
            next_obs_pred = output[..., :self.obs_dim]
            reward_pred = output[..., self.obs_dim:]

        # Apply delta prediction if configured
        if self.predict_delta:
            next_obs_pred = obs + next_obs_pred

        return next_obs_pred, reward_pred

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.

        Args:
            obs: Current observations
            action: Actions
            next_obs: True next observations
            reward: True rewards
            weights: Optional sample weights

        Returns:
            Dictionary of losses
        """
        # Predict
        next_obs_pred, reward_pred = self.forward(obs, action)

        # MSE loss for state prediction
        if self.predict_delta:
            # If predicting delta, compute loss on delta
            delta_true = next_obs - obs
            delta_pred = next_obs_pred - obs
            state_loss = F.mse_loss(delta_pred, delta_true, reduction='none')
        else:
            state_loss = F.mse_loss(next_obs_pred, next_obs, reduction='none')

        # MSE loss for reward prediction
        reward_loss = F.mse_loss(reward_pred, reward.unsqueeze(-1), reduction='none')

        # Apply sample weights if provided
        if weights is not None:
            state_loss = state_loss * weights.unsqueeze(-1)
            reward_loss = reward_loss * weights.unsqueeze(-1)

        # Aggregate losses
        state_loss = state_loss.mean()
        reward_loss = reward_loss.mean()
        total_loss = state_loss + reward_loss

        return {
            "loss": total_loss,
            "state_loss": state_loss,
            "reward_loss": reward_loss,
        }

    def unrolled_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        start_step: int = 0,
        unroll_steps: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss over a multi-step horizon (autoregressive).

        Args:
            obs_seq: Sequence of observations [batch_size, seq_len, obs_dim]
            action_seq: Sequence of actions [batch_size, seq_len, action_dim]
            reward_seq: Sequence of rewards [batch_size, seq_len, 1]
            start_step: Step to start unrolling from
            unroll_steps: Number of steps to unroll

        Returns:
            Dictionary of losses
        """
        batch_size, seq_len, _ = obs_seq.shape
        
        # Ensure we have enough steps
        end_step = min(start_step + unroll_steps, seq_len - 1)
        actual_unroll_steps = end_step - start_step
        
        if actual_unroll_steps <= 0:
            return {"loss": torch.tensor(0.0, device=obs_seq.device), 
                    "state_loss": torch.tensor(0.0, device=obs_seq.device),
                    "reward_loss": torch.tensor(0.0, device=obs_seq.device)}

        total_state_loss = 0
        total_reward_loss = 0
        
        # Initial state
        current_obs = obs_seq[:, start_step]
        
        for t in range(actual_unroll_steps):
            # Get action for current step
            action = action_seq[:, start_step + t]
            
            # Target next state and reward
            target_next_obs = obs_seq[:, start_step + t + 1]
            target_reward = reward_seq[:, start_step + t]
            
            # Predict
            next_obs_pred, reward_pred = self.forward(current_obs, action)
            
            # Compute losses
            if self.predict_delta:
                delta_true = target_next_obs - current_obs
                delta_pred = next_obs_pred - current_obs
                # Note: We use the *predicted* current_obs for delta calculation if t > 0
                # But wait, delta_true should be target_next - target_current? 
                # No, if we are unrolling, we want next_obs_pred to match target_next_obs.
                # If predict_delta is True, the model outputs a delta.
                # The reconstructed next_obs_pred is current_obs + delta_pred.
                # We want this to match target_next_obs.
                # So MSE(next_obs_pred, target_next_obs) is correct.
                # However, for stability, sometimes it's better to supervise the delta directly against 
                # (target_next_obs - target_current_obs). 
                # But here current_obs is the *predicted* state from previous step (except at t=0).
                # So we should just compare the absolute states.
                step_state_loss = F.mse_loss(next_obs_pred, target_next_obs, reduction='none')
            else:
                step_state_loss = F.mse_loss(next_obs_pred, target_next_obs, reduction='none')
                
            step_reward_loss = F.mse_loss(reward_pred, target_reward.unsqueeze(-1), reduction='none')
            
            total_state_loss += step_state_loss.mean()
            total_reward_loss += step_reward_loss.mean()
            
            # Update current observation for next step (autoregressive)
            current_obs = next_obs_pred
            
        # Average over steps
        avg_state_loss = total_state_loss / actual_unroll_steps
        avg_reward_loss = total_reward_loss / actual_unroll_steps
        total_loss = avg_state_loss + avg_reward_loss
        
        return {
            "loss": total_loss,
            "state_loss": avg_state_loss,
            "reward_loss": avg_reward_loss,
        }

    def imagine_trajectory(
        self,
        initial_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Imagine a trajectory by repeatedly applying the model.

        Args:
            initial_obs: Initial observation [batch_size, obs_dim]
            actions: Sequence of actions [batch_size, horizon, action_dim]

        Returns:
            observations: Predicted observations [batch_size, horizon+1, obs_dim]
            rewards: Predicted rewards [batch_size, horizon]
        """
        batch_size, horizon, _ = actions.shape

        observations = [initial_obs]
        rewards = []

        obs = initial_obs
        for t in range(horizon):
            action = actions[:, t]
            next_obs, reward = self.forward(obs, action)

            observations.append(next_obs)
            rewards.append(reward.squeeze(-1))

            obs = next_obs

        observations = torch.stack(observations, dim=1)
        rewards = torch.stack(rewards, dim=1)

        return observations, rewards


class EnsembleWorldModel(nn.Module):
    """
    Ensemble of world models for uncertainty estimation.

    Trains multiple models and aggregates predictions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        **kwargs
    ):
        """
        Initialize ensemble world model.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            ensemble_size: Number of models in ensemble
            **kwargs: Arguments passed to individual models
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Create ensemble of models
        self.models = nn.ModuleList([
            WorldModel(obs_dim, action_dim, **kwargs)
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reduce: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.

        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action [batch_size, action_dim]
            reduce: Reduction method ("none", "mean", "mean_std")

        Returns:
            When reduce="none":
                next_obs: [ensemble_size, batch_size, obs_dim]
                rewards: [ensemble_size, batch_size, 1]
            When reduce="mean":
                next_obs: [batch_size, obs_dim]
                rewards: [batch_size, 1]
            When reduce="mean_std":
                (next_obs_mean, next_obs_std): each [batch_size, obs_dim]
                (rewards_mean, rewards_std): each [batch_size, 1]
        """
        predictions = []
        for model in self.models:
            next_obs, reward = model(obs, action)
            predictions.append((next_obs, reward))

        # Stack predictions
        next_obs_all = torch.stack([p[0] for p in predictions])  # [ensemble_size, batch, obs_dim]
        rewards_all = torch.stack([p[1] for p in predictions])    # [ensemble_size, batch, 1]

        if reduce == "none":
            return next_obs_all, rewards_all
        elif reduce == "mean":
            return next_obs_all.mean(dim=0), rewards_all.mean(dim=0)
        elif reduce == "mean_std":
            next_obs_mean = next_obs_all.mean(dim=0)
            next_obs_std = next_obs_all.std(dim=0)
            rewards_mean = rewards_all.mean(dim=0)
            rewards_std = rewards_all.std(dim=0)
            return (next_obs_mean, next_obs_std), (rewards_mean, rewards_std)
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")

    def forward_with_uncertainty(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns mean and standard deviation of predictions.
        """
        (next_obs_mean, next_obs_std), (rewards_mean, rewards_std) = self.forward(
            obs, action, reduce="mean_std"
        )
        return next_obs_mean, next_obs_std, rewards_mean, rewards_std

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        model_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.

        Args:
            obs: Current observations
            action: Actions
            next_obs: True next observations
            reward: True rewards
            model_idx: Index of model to train (None for all)

        Returns:
            Dictionary of losses
        """
        if model_idx is not None:
            # Train single model
            return self.models[model_idx].loss(obs, action, next_obs, reward)
        else:
            # Train all models
            total_loss = 0
            state_loss = 0
            reward_loss = 0

            for model in self.models:
                losses = model.loss(obs, action, next_obs, reward)
                total_loss += losses["loss"]
                state_loss += losses["state_loss"]
                reward_loss += losses["reward_loss"]

            return {
                "loss": total_loss / self.ensemble_size,
                "state_loss": state_loss / self.ensemble_size,
                "reward_loss": reward_loss / self.ensemble_size,
            }