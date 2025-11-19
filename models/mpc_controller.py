"""
Model Predictive Control (MPC) controller using the learned world model.

Plans actions by simulating futures in the learned model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Callable, Union


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
        lambda_risk: float = 0.0,
        device: str = "cpu",
        encoder: Optional[nn.Module] = None,
        use_latent: bool = False,
        stochastic_rollouts: int = 1,
        use_stochastic: bool = False,
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
            lambda_risk: Risk penalty weight (0=risk-neutral, >0=risk-averse)
            device: Device to run on
            encoder: Encoder network for latent space planning
            use_latent: Whether to plan in latent space
            stochastic_rollouts: Number of rollouts for stochastic models
            use_stochastic: Whether to use stochastic sampling for planning
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
        self.lambda_risk = lambda_risk
        self.device = device
        self.encoder = encoder
        self.use_latent = use_latent
        self.stochastic_rollouts = stochastic_rollouts
        self.use_stochastic = use_stochastic

        # Check if we have an ensemble model
        self.is_ensemble = hasattr(world_model, 'ensemble_size')
        if self.is_ensemble:
            print(f"MPC using ensemble model with {world_model.ensemble_size} members")
            if self.lambda_risk > 0:
                print(f"Risk-sensitive planning enabled with lambda_risk={self.lambda_risk}")

        # Move model to device
        self.world_model = self.world_model.to(device)
        self.world_model.eval()
        
        if self.use_latent and self.encoder is not None:
            self.encoder = self.encoder.to(device)
            self.encoder.eval()

        # Initialize CEM parameters
        self.reset_cem()

    def reset_cem(self):
        """Reset CEM parameters."""
        self.cem_mean = None
        self.cem_std = None

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation if using latent space."""
        if not self.use_latent:
            return obs
        
        if self.encoder is None:
            raise ValueError("Encoder required for latent planning but not provided")
            
        with torch.no_grad():
            # Handle potential extra dimensions if needed
            if len(obs.shape) > 2 and hasattr(self.encoder, 'encode'):
                 # If obs is image-like [B, C, H, W]
                 return self.encoder.encode(obs)
            return self.encoder(obs)

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
        
        # Encode if needed
        state_tensor = self._encode_obs(obs_tensor.unsqueeze(0)).squeeze(0)

        # Plan
        if self.use_cem:
            action, info = self._plan_cem(state_tensor)
        else:
            action, info = self._plan_random_shooting(state_tensor)

        # Convert to numpy
        action = action.cpu().numpy()

        if return_info:
            return action, info
        return action

    def _plan_random_shooting(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Plan using random shooting.

        Args:
            state: Current state (observation or latent) [state_dim]

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

        # Evaluate sequences (may include risk adjustment)
        returns = self._evaluate_sequences(state, action_sequences)

        # Select best sequence (based on risk-adjusted returns if applicable)
        best_idx = returns.argmax()
        best_sequence = action_sequences[best_idx]
        best_return = returns[best_idx].item()

        info = {
            "best_return": best_return,
            "mean_return": returns.mean().item(),
            "std_return": returns.std().item(),
        }

        # Add risk info if applicable
        if (self.is_ensemble or self.use_stochastic) and self.lambda_risk > 0:
            info["risk_penalty"] = self.lambda_risk
            info["planning_mode"] = "risk-sensitive"
        else:
            info["planning_mode"] = "risk-neutral"

        return best_sequence[0], info

    def _plan_cem(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Plan using Cross-Entropy Method.

        Args:
            state: Current state (observation or latent) [state_dim]

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
        
        # Track returns for info
        all_returns = []

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
            returns = self._evaluate_sequences(state, action_sequences)
            all_returns = returns

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
            "mean_return": all_returns.mean().item(),
            "std_return": all_returns.std().item(),
            "optimization_iters": self.optimization_iters,
        }

        # Add risk info if applicable
        if (self.is_ensemble or self.use_stochastic) and self.lambda_risk > 0:
            info["risk_penalty"] = self.lambda_risk
            info["planning_mode"] = "risk-sensitive"
        else:
            info["planning_mode"] = "risk-neutral"

        return best_sequence[0], info

    def _evaluate_sequences(
        self,
        state: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate action sequences using the world model.

        Args:
            state: Initial state [state_dim]
            action_sequences: Action sequences [num_samples, horizon, action_dim]

        Returns:
            returns: Risk-adjusted returns for each sequence [num_samples]
        """
        num_samples = action_sequences.shape[0]

        # Expand initial state for all samples
        state_expanded = state.unsqueeze(0).repeat(num_samples, 1)

        if self.is_ensemble and self.lambda_risk > 0:
            # For ensemble with risk-sensitive planning, track returns per member
            returns_per_member = torch.zeros(
                self.world_model.ensemble_size,
                num_samples,
                device=self.device
            )

            # Roll out each sequence with each ensemble member
            with torch.no_grad():
                for member_idx in range(self.world_model.ensemble_size):
                    current_state = state_expanded
                    discount = 1.0

                    for t in range(self.horizon):
                        actions = action_sequences[:, t]

                        # Predict with specific ensemble member
                        next_state, rewards = self.world_model.models[member_idx](current_state, actions)

                        # Accumulate discounted reward
                        returns_per_member[member_idx] += discount * rewards.squeeze(-1)
                        discount *= self.gamma

                        # Update state
                        current_state = next_state

            # Compute mean and std of returns across ensemble
            mean_returns = returns_per_member.mean(dim=0)
            std_returns = returns_per_member.std(dim=0)

            # Risk-sensitive scoring: mean - lambda * std
            returns = mean_returns - self.lambda_risk * std_returns

        elif self.use_stochastic:
            # Stochastic rollouts (Workstream 2)
            # We need to repeat samples for stochastic rollouts
            # [num_samples * stochastic_rollouts, state_dim]
            batch_size = num_samples * self.stochastic_rollouts
            
            current_state = state.unsqueeze(0).repeat(batch_size, 1)
            
            # Expand actions: [num_samples, horizon, dim] -> [num_samples, rollouts, horizon, dim] -> [batch, horizon, dim]
            actions_expanded = action_sequences.unsqueeze(1).repeat(1, self.stochastic_rollouts, 1, 1)
            actions_expanded = actions_expanded.view(batch_size, self.horizon, self.action_dim)
            
            total_returns = torch.zeros(batch_size, device=self.device)
            discount = 1.0
            
            with torch.no_grad():
                for t in range(self.horizon):
                    actions = actions_expanded[:, t]
                    
                    # Predict (stochastic sampling should happen inside model if not deterministic)
                    # Assuming model.forward samples if sample=True
                    # Check signature of forward
                    import inspect
                    forward_args = inspect.signature(self.world_model.forward).parameters
                    if 'sample' in forward_args:
                         next_state, rewards, *extras = self.world_model(current_state, actions, sample=True)
                    elif 'deterministic' in forward_args:
                         next_state, rewards, *extras = self.world_model(current_state, actions, deterministic=False)
                    else:
                         # Fallback for models that don't support explicit sampling control
                         next_state, rewards = self.world_model(current_state, actions)
                    
                    # Handle potential extra outputs (like dist params)
                    if isinstance(rewards, tuple):
                        rewards = rewards[0]
                        
                    total_returns += discount * rewards.squeeze(-1)
                    discount *= self.gamma
                    current_state = next_state
            
            # Reshape to [num_samples, rollouts]
            returns_matrix = total_returns.view(num_samples, self.stochastic_rollouts)
            
            mean_returns = returns_matrix.mean(dim=1)
            std_returns = returns_matrix.std(dim=1)
            
            if self.lambda_risk > 0:
                returns = mean_returns - self.lambda_risk * std_returns
            else:
                returns = mean_returns

        else:
            # Standard evaluation (single model or risk-neutral ensemble)
            returns = torch.zeros(num_samples, device=self.device)
            discount = 1.0

            # Roll out each sequence
            with torch.no_grad():
                current_state = state_expanded

                for t in range(self.horizon):
                    actions = action_sequences[:, t]

                    # Predict next state and reward
                    if self.is_ensemble:
                        # Use mean predictions for risk-neutral ensemble
                        next_state, rewards = self.world_model(current_state, actions, reduce="mean")
                    else:
                        # Deterministic prediction
                        # If model is stochastic but use_stochastic=False, we want mean (deterministic).
                        
                        # Check if model accepts deterministic/sample arg
                        import inspect
                        forward_args = inspect.signature(self.world_model.forward).parameters
                        if 'sample' in forward_args:
                             next_state, rewards, *extras = self.world_model(current_state, actions, sample=False)
                        elif 'deterministic' in forward_args:
                             next_state, rewards, *extras = self.world_model(current_state, actions, deterministic=True)
                        else:
                             next_state, rewards = self.world_model(current_state, actions)
                             
                        # Handle potential extra outputs
                        if isinstance(rewards, tuple):
                            rewards = rewards[0]

                    # Accumulate discounted reward
                    returns += discount * rewards.squeeze(-1)
                    discount *= self.gamma

                    # Update state
                    current_state = next_state

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

        obs_tensor = torch.from_numpy(initial_obs).float().to(self.device)
        
        # Encode if needed
        state = self._encode_obs(obs_tensor.unsqueeze(0)).squeeze(0)

        # Get best action sequence
        if self.use_cem:
            # Run CEM
            self.reset_cem()
            _, _ = self._plan_cem(state)

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

            returns = self._evaluate_sequences(state, action_sequences)
            best_idx = returns.argmax()
            action_sequence = action_sequences[best_idx]

        # Roll out best sequence
        states = [state.cpu().numpy()]
        rewards = []

        with torch.no_grad():
            current_state = state.unsqueeze(0)

            for t in range(horizon):
                action = action_sequence[t].unsqueeze(0)

                # Predict
                if self.is_ensemble:
                    next_state, reward = self.world_model(current_state, action, reduce="mean")
                else:
                    # Use deterministic for visualization
                    import inspect
                    forward_args = inspect.signature(self.world_model.forward).parameters
                    if 'sample' in forward_args:
                         next_state, reward, *extras = self.world_model(current_state, action, sample=False)
                    elif 'deterministic' in forward_args:
                         next_state, reward, *extras = self.world_model(current_state, action, deterministic=True)
                    else:
                         next_state, reward = self.world_model(current_state, action)
                    
                    if isinstance(reward, tuple):
                        reward = reward[0]

                # Store
                states.append(next_state.squeeze(0).cpu().numpy())
                rewards.append(reward.item())

                # Update
                current_state = next_state

        actions = action_sequence.cpu().numpy()
        states = np.stack(states)
        rewards = np.array(rewards)
        
        return actions, states, rewards


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