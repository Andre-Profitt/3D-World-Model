"""
Model Predictive Control (MPC) controller using the learned world model.

Plans actions by simulating futures in the learned model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Callable


class MPCController:
    """
    MPC controller using random shooting or CEM.

    Samples action sequences, evaluates them using the world model,
    and selects the best one.
    """

    def __init__(
        self,
        world_model: nn.Module,
        action_dim: int,
        horizon: int = 15,
        num_samples: int = 1024,
        num_elite: int = 64,
        gamma: float = 0.99,
        temperature: float = 0.1,
        action_noise: float = 0.3,
        optimization_iters: int = 3,
        use_cem: bool = True,
        action_min: float = -5.0,
        action_max: float = 5.0,
        device: str = "cpu",
    ):
        """
        Initialize MPC controller.

        Args:
            world_model: Trained world model for dynamics prediction
            action_dim: Dimension of action space
            horizon: Planning horizon
            num_samples: Number of action sequences to sample
            num_elite: Number of elite samples for CEM
            gamma: Discount factor for rewards
            temperature: Temperature for action sampling
            action_noise: Noise level for action sampling
            optimization_iters: Number of CEM optimization iterations
            use_cem: Whether to use CEM or random shooting
            action_min: Minimum action value
            action_max: Maximum action value
            device: Device to run on
        """
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elite = num_elite
        self.gamma = gamma
        self.temperature = temperature
        self.action_noise = action_noise
        self.optimization_iters = optimization_iters
        self.use_cem = use_cem
        self.action_min = action_min
        self.action_max = action_max
        self.device = device

        # Move model to device
        self.world_model = self.world_model.to(device)
        self.world_model.eval()

        # Initialize CEM parameters
        self.reset_cem()

    def reset_cem(self):
        """Reset CEM parameters."""
        self.cem_mean = None
        self.cem_std = None

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        return_info: bool = False,
    ) -> np.ndarray:
        """
        Select action using MPC.

        Args:
            obs: Current observation
            deterministic: Whether to use deterministic planning
            return_info: Whether to return additional planning info

        Returns:
            action: Selected action
        """
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        # Plan
        if self.use_cem:
            action, info = self._plan_cem(obs_tensor)
        else:
            action, info = self._plan_random_shooting(obs_tensor)

        # Convert to numpy
        action = action.cpu().numpy()

        if return_info:
            return action, info
        return action

    def _plan_random_shooting(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Plan using random shooting.

        Args:
            obs: Current observation [obs_dim]

        Returns:
            action: Best first action
            info: Planning information
        """
        # Sample random action sequences
        action_sequences = torch.uniform_(
            torch.zeros(
                self.num_samples,
                self.horizon,
                self.action_dim,
                device=self.device
            ),
            self.action_min,
            self.action_max
        )

        # Evaluate sequences
        returns = self._evaluate_sequences(obs, action_sequences)

        # Select best sequence
        best_idx = returns.argmax()
        best_sequence = action_sequences[best_idx]
        best_return = returns[best_idx].item()

        info = {
            "best_return": best_return,
            "mean_return": returns.mean().item(),
            "std_return": returns.std().item(),
        }

        return best_sequence[0], info

    def _plan_cem(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Plan using Cross-Entropy Method.

        Args:
            obs: Current observation [obs_dim]

        Returns:
            action: Best first action
            info: Planning information
        """
        # Initialize CEM distribution
        if self.cem_mean is None:
            self.cem_mean = torch.zeros(
                self.horizon,
                self.action_dim,
                device=self.device
            )
            self.cem_std = torch.ones(
                self.horizon,
                self.action_dim,
                device=self.device
            ) * self.action_noise

        best_return = -float("inf")
        best_sequence = None

        for iter_idx in range(self.optimization_iters):
            # Sample action sequences from current distribution
            noise = torch.randn(
                self.num_samples,
                self.horizon,
                self.action_dim,
                device=self.device
            )
            action_sequences = self.cem_mean + self.cem_std * noise

            # Clip actions
            action_sequences = torch.clamp(
                action_sequences,
                self.action_min,
                self.action_max
            )

            # Evaluate sequences
            returns = self._evaluate_sequences(obs, action_sequences)

            # Select elite sequences
            elite_indices = returns.argsort(descending=True)[:self.num_elite]
            elite_sequences = action_sequences[elite_indices]
            elite_returns = returns[elite_indices]

            # Update distribution
            self.cem_mean = elite_sequences.mean(dim=0)
            self.cem_std = elite_sequences.std(dim=0) + 1e-6

            # Track best
            if elite_returns[0] > best_return:
                best_return = elite_returns[0].item()
                best_sequence = elite_sequences[0]

        # Shift mean for next timestep (receding horizon)
        self.cem_mean[:-1] = self.cem_mean[1:].clone()
        self.cem_mean[-1] = 0

        info = {
            "best_return": best_return,
            "mean_return": returns.mean().item(),
            "std_return": returns.std().item(),
            "optimization_iters": self.optimization_iters,
        }

        return best_sequence[0], info

    def _evaluate_sequences(
        self,
        obs: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate action sequences using the world model.

        Args:
            obs: Initial observation [obs_dim]
            action_sequences: Action sequences [num_samples, horizon, action_dim]

        Returns:
            returns: Discounted returns for each sequence [num_samples]
        """
        num_samples = action_sequences.shape[0]

        # Expand initial observation for all samples
        obs_expanded = obs.unsqueeze(0).repeat(num_samples, 1)

        returns = torch.zeros(num_samples, device=self.device)
        discount = 1.0

        # Roll out each sequence
        with torch.no_grad():
            current_obs = obs_expanded

            for t in range(self.horizon):
                actions = action_sequences[:, t]

                # Predict next state and reward
                next_obs, rewards = self.world_model(current_obs, actions)

                # Accumulate discounted reward
                returns += discount * rewards.squeeze(-1)
                discount *= self.gamma

                # Update observation
                current_obs = next_obs

        return returns

    def plan_trajectory(
        self,
        initial_obs: np.ndarray,
        horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan full trajectory from initial observation.

        Args:
            initial_obs: Initial observation
            horizon: Planning horizon (default: self.horizon)

        Returns:
            actions: Planned action sequence [horizon, action_dim]
            observations: Predicted observations [horizon+1, obs_dim]
            rewards: Predicted rewards [horizon]
        """
        if horizon is None:
            horizon = self.horizon

        obs = torch.from_numpy(initial_obs).float().to(self.device)

        # Get best action sequence
        if self.use_cem:
            # Run CEM
            self.reset_cem()
            _, _ = self._plan_cem(obs)

            # Best sequence is in cem_mean
            action_sequence = self.cem_mean[:horizon].clone()
        else:
            # Run random shooting
            action_sequences = torch.uniform_(
                torch.zeros(
                    self.num_samples,
                    horizon,
                    self.action_dim,
                    device=self.device
                ),
                self.action_min,
                self.action_max
            )

            returns = self._evaluate_sequences(obs, action_sequences)
            best_idx = returns.argmax()
            action_sequence = action_sequences[best_idx]

        # Roll out best sequence
        observations = [obs.cpu().numpy()]
        rewards = []

        with torch.no_grad():
            current_obs = obs.unsqueeze(0)

            for t in range(horizon):
                action = action_sequence[t].unsqueeze(0)

                # Predict
                next_obs, reward = self.world_model(current_obs, action)

                # Store
                observations.append(next_obs.squeeze(0).cpu().numpy())
                rewards.append(reward.item())

                # Update
                current_obs = next_obs

        actions = action_sequence.cpu().numpy()
        observations = np.stack(observations)
        rewards = np.array(rewards)

        return actions, observations, rewards


class MPCAgent:
    """
    Agent that uses MPC for control.

    Wraps the MPC controller with additional functionality
    like warm-starting and action smoothing.
    """

    def __init__(
        self,
        controller: MPCController,
        action_smoothing: float = 0.0,
        warm_start: bool = True,
    ):
        """
        Initialize MPC agent.

        Args:
            controller: MPC controller instance
            action_smoothing: Smoothing factor for actions (0=no smoothing)
            warm_start: Whether to warm-start CEM from previous solution
        """
        self.controller = controller
        self.action_smoothing = action_smoothing
        self.warm_start = warm_start

        self.prev_action = None

    def reset(self):
        """Reset agent state."""
        self.prev_action = None
        if not self.warm_start:
            self.controller.reset_cem()

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action.

        Args:
            obs: Current observation
            deterministic: Whether to use deterministic planning

        Returns:
            action: Selected action
        """
        # Get action from MPC
        action = self.controller.select_action(obs, deterministic)

        # Apply action smoothing
        if self.action_smoothing > 0 and self.prev_action is not None:
            action = (
                self.action_smoothing * self.prev_action +
                (1 - self.action_smoothing) * action
            )

        self.prev_action = action.copy()
        return action