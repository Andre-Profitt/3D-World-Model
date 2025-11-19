"""
Stochastic World Model with probabilistic dynamics.

Outputs distributions over next states and rewards for uncertainty-aware planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal, kl_divergence
from typing import Tuple, Dict, Optional, List
import numpy as np


class StochasticMLP(nn.Module):
    """MLP that outputs mean and log variance for a distribution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
        min_std: float = 0.01,
        max_std: float = 1.0,
    ):
        """
        Initialize stochastic MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (for mean, logvar will be same)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
        """
        super().__init__()

        self.output_dim = output_dim
        self.min_std = min_std
        self.max_std = max_std

        # Build shared layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Last hidden layer (shared)
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "elu":
            layers.append(nn.ELU())

        self.shared_layers = nn.Sequential(*layers)

        # Separate heads for mean and log variance
        self.mean_head = nn.Linear(hidden_dims[-1], output_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get distribution parameters.

        Args:
            x: Input tensor

        Returns:
            mean: Mean of the distribution
            std: Standard deviation of the distribution
        """
        # Shared computation
        features = self.shared_layers(x)

        # Get mean and log variance
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        # Convert to standard deviation with bounds
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, self.min_std, self.max_std)

        return mean, std


class StochasticWorldModel(nn.Module):
    """
    Stochastic world model with probabilistic dynamics.

    Predicts distributions over next states and rewards.
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
        min_std: float = 0.01,
        max_std: float = 1.0,
        separate_reward_head: bool = True,
        deterministic: bool = False,
    ):
        """
        Initialize stochastic world model.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
            predict_delta: Whether to predict state changes
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
            separate_reward_head: Use separate network for rewards
            deterministic: If True, always return mean (no sampling)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.predict_delta = predict_delta
        self.separate_reward_head = separate_reward_head
        self.deterministic = deterministic

        # Input: concatenated observation and action
        input_dim = obs_dim + action_dim

        # Dynamics network (outputs distribution over next state)
        self.dynamics_net = StochasticMLP(
            input_dim=input_dim,
            output_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            min_std=min_std,
            max_std=max_std,
        )

        if separate_reward_head:
            # Separate reward network
            self.reward_net = StochasticMLP(
                input_dim=input_dim,
                output_dim=1,
                hidden_dims=[hidden_dims[0], hidden_dims[0] // 2],
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
                min_std=min_std,
                max_std=max_std,
            )
        else:
            # Joint network for state and reward
            self.joint_net = StochasticMLP(
                input_dim=input_dim,
                output_dim=obs_dim + 1,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                layer_norm=layer_norm,
                min_std=min_std,
                max_std=max_std,
            )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        deterministic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with stochastic dynamics.

        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action [batch_size, action_dim]
            deterministic: Override deterministic mode

        Returns:
            next_obs: Sampled next observation
            reward: Sampled reward
            dist_params: Distribution parameters (means and stds)
        """
        if deterministic is None:
            deterministic = self.deterministic

        # Concatenate inputs
        x = torch.cat([obs, action], dim=-1)

        if self.separate_reward_head:
            # Get state distribution
            state_mean, state_std = self.dynamics_net(x)

            # Get reward distribution
            reward_mean, reward_std = self.reward_net(x)
        else:
            # Joint prediction
            joint_mean, joint_std = self.joint_net(x)
            state_mean = joint_mean[..., :self.obs_dim]
            state_std = joint_std[..., :self.obs_dim]
            reward_mean = joint_mean[..., self.obs_dim:]
            reward_std = joint_std[..., self.obs_dim:]

        # Apply delta prediction if configured
        if self.predict_delta:
            state_mean = obs + state_mean

        # Sample or use mean
        if deterministic:
            next_obs = state_mean
            reward = reward_mean
        else:
            # Sample from distributions
            state_dist = Normal(state_mean, state_std)
            reward_dist = Normal(reward_mean, reward_std)

            next_obs = state_dist.rsample()  # Reparameterization trick
            reward = reward_dist.rsample()

        # Store distribution parameters
        dist_params = {
            "state_mean": state_mean,
            "state_std": state_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
        }

        return next_obs, reward, dist_params

    def get_distribution(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[Normal, Normal]:
        """
        Get distributions without sampling.

        Args:
            obs: Current observation
            action: Action

        Returns:
            state_dist: Distribution over next states
            reward_dist: Distribution over rewards
        """
        x = torch.cat([obs, action], dim=-1)

        if self.separate_reward_head:
            state_mean, state_std = self.dynamics_net(x)
            reward_mean, reward_std = self.reward_net(x)
        else:
            joint_mean, joint_std = self.joint_net(x)
            state_mean = joint_mean[..., :self.obs_dim]
            state_std = joint_std[..., :self.obs_dim]
            reward_mean = joint_mean[..., self.obs_dim:]
            reward_std = joint_std[..., self.obs_dim:]

        if self.predict_delta:
            state_mean = obs + state_mean

        state_dist = Normal(state_mean, state_std)
        reward_dist = Normal(reward_mean, reward_std)

        return state_dist, reward_dist

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute negative log-likelihood loss.

        Args:
            obs: Current observations
            action: Actions
            next_obs: True next observations
            reward: True rewards
            beta: Weight for KL regularization (if using)

        Returns:
            Dictionary of losses
        """
        # Get distributions
        state_dist, reward_dist = self.get_distribution(obs, action)

        # Negative log-likelihood for states
        if self.predict_delta:
            delta_true = next_obs - obs
            delta_pred = state_dist.mean - obs
            state_nll = -state_dist.log_prob(next_obs).mean()
        else:
            state_nll = -state_dist.log_prob(next_obs).mean()

        # Negative log-likelihood for rewards
        reward_nll = -reward_dist.log_prob(reward.unsqueeze(-1)).mean()

        # Total loss
        total_loss = state_nll + reward_nll

        # Optional: Add entropy regularization to prevent collapse
        state_entropy = state_dist.entropy().mean()
        reward_entropy = reward_dist.entropy().mean()
        entropy_reg = -0.01 * (state_entropy + reward_entropy)

        total_loss = total_loss + entropy_reg

        return {
            "loss": total_loss,
            "state_nll": state_nll,
            "reward_nll": reward_nll,
            "state_entropy": state_entropy,
            "reward_entropy": reward_entropy,
        }

    def imagine_trajectory(
        self,
        initial_obs: torch.Tensor,
        actions: torch.Tensor,
        num_particles: int = 1,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine trajectories with uncertainty propagation.

        Args:
            initial_obs: Initial observation [batch_size, obs_dim]
            actions: Action sequence [batch_size, horizon, action_dim]
            num_particles: Number of stochastic particles
            deterministic: Use mean instead of sampling

        Returns:
            Dictionary with trajectories and uncertainty estimates
        """
        batch_size, horizon, _ = actions.shape

        # Expand for particles
        if num_particles > 1 and not deterministic:
            initial_obs = initial_obs.unsqueeze(1).repeat(1, num_particles, 1)
            initial_obs = initial_obs.view(batch_size * num_particles, -1)
            actions = actions.unsqueeze(1).repeat(1, num_particles, 1, 1)
            actions = actions.view(batch_size * num_particles, horizon, -1)

        observations = [initial_obs]
        rewards = []
        state_means = []
        state_stds = []
        reward_means = []
        reward_stds = []

        obs = initial_obs
        for t in range(horizon):
            action = actions[:, t] if num_particles > 1 else actions[:, t]

            next_obs, reward, dist_params = self.forward(
                obs, action, deterministic=deterministic
            )

            observations.append(next_obs)
            rewards.append(reward.squeeze(-1))
            state_means.append(dist_params["state_mean"])
            state_stds.append(dist_params["state_std"])
            reward_means.append(dist_params["reward_mean"])
            reward_stds.append(dist_params["reward_std"])

            obs = next_obs

        # Stack temporal dimension
        observations = torch.stack(observations, dim=1)
        rewards = torch.stack(rewards, dim=1)
        state_means = torch.stack(state_means, dim=1)
        state_stds = torch.stack(state_stds, dim=1)
        reward_means = torch.stack(reward_means, dim=1)
        reward_stds = torch.stack(reward_stds, dim=1)

        # Reshape particles back
        if num_particles > 1 and not deterministic:
            observations = observations.view(batch_size, num_particles, horizon + 1, -1)
            rewards = rewards.view(batch_size, num_particles, horizon)
            state_means = state_means.view(batch_size, num_particles, horizon, -1)
            state_stds = state_stds.view(batch_size, num_particles, horizon, -1)
            reward_means = reward_means.view(batch_size, num_particles, horizon, -1)
            reward_stds = reward_stds.view(batch_size, num_particles, horizon, -1)

            # Compute statistics across particles
            obs_mean = observations.mean(dim=1)
            obs_std = observations.std(dim=1)
            reward_particle_mean = rewards.mean(dim=1)
            reward_particle_std = rewards.std(dim=1)
        else:
            obs_mean = observations
            obs_std = state_stds
            reward_particle_mean = rewards
            reward_particle_std = reward_stds

        return {
            "observations": observations,
            "rewards": rewards,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "reward_mean": reward_particle_mean,
            "reward_std": reward_particle_std,
            "state_means": state_means,
            "state_stds": state_stds,
            "reward_means": reward_means,
            "reward_stds": reward_stds,
        }


class StochasticEnsembleWorldModel(nn.Module):
    """
    Ensemble of stochastic world models.

    Combines epistemic uncertainty (ensemble) with aleatoric uncertainty (stochastic).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        **kwargs
    ):
        """
        Initialize stochastic ensemble.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            ensemble_size: Number of models in ensemble
            **kwargs: Arguments for individual models
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Create ensemble of stochastic models
        self.models = nn.ModuleList([
            StochasticWorldModel(obs_dim, action_dim, **kwargs)
            for _ in range(ensemble_size)
        ])

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False,
        reduce: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through ensemble.

        Args:
            obs: Current observation
            action: Action
            deterministic: Use mean instead of sampling
            reduce: Reduction over ensemble ("none", "mean", "mean_std")

        Returns:
            Predictions and uncertainty estimates
        """
        predictions = []
        dist_params_list = []

        for model in self.models:
            next_obs, reward, dist_params = model(obs, action, deterministic)
            predictions.append((next_obs, reward))
            dist_params_list.append(dist_params)

        # Stack predictions
        next_obs_all = torch.stack([p[0] for p in predictions])
        rewards_all = torch.stack([p[1] for p in predictions])

        if reduce == "none":
            return next_obs_all, rewards_all, dist_params_list
        elif reduce == "mean":
            return next_obs_all.mean(dim=0), rewards_all.mean(dim=0), dist_params_list
        elif reduce == "mean_std":
            # Epistemic uncertainty from ensemble
            next_obs_mean = next_obs_all.mean(dim=0)
            next_obs_epistemic_std = next_obs_all.std(dim=0)
            rewards_mean = rewards_all.mean(dim=0)
            rewards_epistemic_std = rewards_all.std(dim=0)

            # Aleatoric uncertainty (average from models)
            state_aleatoric_std = torch.stack([d["state_std"] for d in dist_params_list]).mean(dim=0)
            reward_aleatoric_std = torch.stack([d["reward_std"] for d in dist_params_list]).mean(dim=0)

            # Total uncertainty (approximate)
            next_obs_total_std = torch.sqrt(next_obs_epistemic_std**2 + state_aleatoric_std**2)
            rewards_total_std = torch.sqrt(rewards_epistemic_std**2 + reward_aleatoric_std**2)

            uncertainty_info = {
                "epistemic_state_std": next_obs_epistemic_std,
                "aleatoric_state_std": state_aleatoric_std,
                "total_state_std": next_obs_total_std,
                "epistemic_reward_std": rewards_epistemic_std,
                "aleatoric_reward_std": reward_aleatoric_std,
                "total_reward_std": rewards_total_std,
            }

            return (next_obs_mean, rewards_mean), uncertainty_info, dist_params_list
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")


class StochasticMPCWrapper(nn.Module):
    """
    Wrapper for stochastic world model to work with standard MPC controller.
    
    Strips the distribution parameters from the output.
    """
    
    def __init__(self, model: StochasticWorldModel):
        super().__init__()
        self.model = model
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning only next_obs and reward."""
        next_obs, reward, _ = self.model(obs, action)
        return next_obs, reward