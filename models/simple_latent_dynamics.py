"""
Simple latent dynamics model for VAE-encoded states.
"""

import torch
import torch.nn as nn


class SimpleLatentDynamics(nn.Module):
    """
    Simple MLP dynamics model for latent space transitions.

    Predicts:
    - Next latent state
    - Reward
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = 'relu',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Input is concatenated [z_t, a_t]
        input_dim = latent_dim + action_dim

        # Build dynamics network
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

        self.dynamics_net = nn.Sequential(*layers)

        # Output heads
        self.next_state_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Predict next state and reward.

        Args:
            state: Current latent state [batch_size, latent_dim]
            action: Action [batch_size, action_dim]

        Returns:
            next_state: Predicted next latent state [batch_size, latent_dim]
            reward: Predicted reward [batch_size, 1]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Pass through dynamics network
        h = self.dynamics_net(x)

        # Predict next state (as delta/residual)
        delta_state = self.next_state_head(h)
        next_state = state + delta_state  # Residual connection

        # Predict reward
        reward = self.reward_head(h)

        return next_state, reward

    def rollout(self, initial_state: torch.Tensor, actions: torch.Tensor):
        """
        Rollout dynamics for a sequence of actions.

        Args:
            initial_state: Initial latent state [batch_size, latent_dim]
            actions: Sequence of actions [batch_size, horizon, action_dim]

        Returns:
            states: Predicted states [batch_size, horizon + 1, latent_dim]
            rewards: Predicted rewards [batch_size, horizon]
        """
        batch_size, horizon, _ = actions.shape

        states = [initial_state]
        rewards = []

        state = initial_state
        for t in range(horizon):
            action = actions[:, t]
            next_state, reward = self.forward(state, action)
            states.append(next_state)
            rewards.append(reward)
            state = next_state

        states = torch.stack(states, dim=1)
        rewards = torch.stack(rewards, dim=1).squeeze(-1)

        return states, rewards