"""
Risk metrics for uncertainty-aware planning.

Implements various risk measures for robust decision making under uncertainty.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class RiskMetrics:
    """Collection of risk metrics for trajectory evaluation."""

    @staticmethod
    def cvar(
        values: torch.Tensor,
        alpha: float = 0.1,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute Conditional Value at Risk (CVaR).

        CVaR is the expected value of the worst alpha-fraction of outcomes.

        Args:
            values: Tensor of values (e.g., returns)
            alpha: Risk level (0.1 = 10% worst cases)
            dim: Dimension along which to compute CVaR

        Returns:
            CVaR values
        """
        # Sort values along specified dimension
        sorted_values, _ = torch.sort(values, dim=dim)

        # Get number of samples to consider
        n_samples = values.shape[dim]
        n_worst = max(1, int(alpha * n_samples))

        # Take worst alpha-fraction
        if dim == -1 or dim == len(values.shape) - 1:
            worst_values = sorted_values[..., :n_worst]
        elif dim == 0:
            worst_values = sorted_values[:n_worst, ...]
        elif dim == 1:
            worst_values = sorted_values[:, :n_worst, ...]
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        # Return mean of worst cases
        return torch.mean(worst_values, dim=dim)

    @staticmethod
    def var(
        values: torch.Tensor,
        alpha: float = 0.1,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute Value at Risk (VaR).

        VaR is the alpha-quantile of the distribution.

        Args:
            values: Tensor of values
            alpha: Risk level
            dim: Dimension along which to compute VaR

        Returns:
            VaR values
        """
        quantile = alpha
        return torch.quantile(values, quantile, dim=dim)

    @staticmethod
    def worst_case(
        values: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute worst-case value.

        Args:
            values: Tensor of values
            dim: Dimension along which to find worst case

        Returns:
            Worst-case values
        """
        return torch.min(values, dim=dim)[0]

    @staticmethod
    def mean_std_penalty(
        mean: torch.Tensor,
        std: torch.Tensor,
        lambda_risk: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute risk-penalized value using mean-std formula.

        Args:
            mean: Mean values
            std: Standard deviations
            lambda_risk: Risk aversion parameter

        Returns:
            Risk-adjusted values
        """
        return mean - lambda_risk * std

    @staticmethod
    def entropic_risk(
        values: torch.Tensor,
        beta: float = 1.0,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Compute entropic risk measure.

        Args:
            values: Tensor of values
            beta: Risk parameter (higher = more risk-averse)
            dim: Dimension for reduction

        Returns:
            Entropic risk values
        """
        # Numerical stability
        max_val = torch.max(values, dim=dim, keepdim=True)[0]
        adjusted_values = values - max_val

        # Entropic risk
        exp_values = torch.exp(-beta * adjusted_values)
        risk = -(1.0 / beta) * torch.log(torch.mean(exp_values, dim=dim))
        risk = risk + max_val.squeeze(dim)

        return risk


class RiskSensitiveMPC:
    """
    Risk-sensitive Model Predictive Control.

    Extends MPC with risk metrics for robust planning.
    """

    def __init__(
        self,
        world_model: torch.nn.Module,
        horizon: int = 15,
        num_samples: int = 1024,
        num_particles: int = 10,
        risk_metric: str = "cvar",
        risk_level: float = 0.1,
        lambda_risk: float = 1.0,
        temperature: float = 0.1,
        use_cem: bool = True,
        num_elite: int = 64,
        optimization_iters: int = 3,
    ):
        """
        Initialize risk-sensitive MPC.

        Args:
            world_model: Stochastic world model
            horizon: Planning horizon
            num_samples: Number of action sequences to sample
            num_particles: Number of stochastic particles per sequence
            risk_metric: Risk metric to use ("cvar", "var", "worst_case", "mean_std")
            risk_level: Risk level for CVaR/VaR
            lambda_risk: Risk aversion parameter
            temperature: Temperature for action sampling
            use_cem: Use Cross-Entropy Method
            num_elite: Number of elite samples for CEM
            optimization_iters: CEM optimization iterations
        """
        self.world_model = world_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_particles = num_particles
        self.risk_metric = risk_metric
        self.risk_level = risk_level
        self.lambda_risk = lambda_risk
        self.temperature = temperature
        self.use_cem = use_cem
        self.num_elite = num_elite
        self.optimization_iters = optimization_iters

        self.risk_metrics = RiskMetrics()

    def evaluate_action_sequences(
        self,
        obs: torch.Tensor,
        action_sequences: torch.Tensor,
        gamma: float = 0.99,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Evaluate action sequences with risk considerations.

        Args:
            obs: Current observation [batch_size, obs_dim]
            action_sequences: Actions [num_samples, horizon, action_dim]
            gamma: Discount factor

        Returns:
            risk_adjusted_returns: Risk-adjusted returns for each sequence
            info: Dictionary with additional metrics
        """
        num_samples = action_sequences.shape[0]
        batch_size = obs.shape[0]
        device = obs.device

        # Expand observations for all samples
        expanded_obs = obs.unsqueeze(0).repeat(num_samples, 1, 1)
        expanded_obs = expanded_obs.view(num_samples * batch_size, -1)

        # Expand actions for particles
        if self.num_particles > 1:
            action_sequences = action_sequences.unsqueeze(1).repeat(
                1, self.num_particles, 1, 1
            )
            action_sequences = action_sequences.view(
                num_samples * self.num_particles, self.horizon, -1
            )
            expanded_obs = expanded_obs.unsqueeze(1).repeat(1, self.num_particles, 1)
            expanded_obs = expanded_obs.view(
                num_samples * self.num_particles * batch_size, -1
            )

        # Imagine trajectories with uncertainty
        with torch.no_grad():
            if hasattr(self.world_model, "imagine_trajectory"):
                trajectory = self.world_model.imagine_trajectory(
                    expanded_obs,
                    action_sequences,
                    num_particles=1,  # Particles handled externally
                    deterministic=False,
                )
                rewards = trajectory["rewards"]

                # Get uncertainty estimates if available
                if "reward_std" in trajectory:
                    reward_stds = trajectory["reward_std"]
                else:
                    reward_stds = torch.zeros_like(rewards)
            else:
                # Manual rollout for models without imagine_trajectory
                rewards = []
                reward_stds = []
                current_obs = expanded_obs

                for t in range(self.horizon):
                    action = action_sequences[:, t]

                    # Get stochastic predictions
                    next_obs, reward, dist_params = self.world_model(
                        current_obs, action, deterministic=False
                    )

                    rewards.append(reward.squeeze(-1))

                    if "reward_std" in dist_params:
                        reward_stds.append(dist_params["reward_std"].squeeze(-1))
                    else:
                        reward_stds.append(torch.zeros_like(reward.squeeze(-1)))

                    current_obs = next_obs

                rewards = torch.stack(rewards, dim=1)
                reward_stds = torch.stack(reward_stds, dim=1)

        # Reshape rewards if using particles
        if self.num_particles > 1:
            rewards = rewards.view(
                num_samples, self.num_particles, batch_size, self.horizon
            )
            reward_stds = reward_stds.view(
                num_samples, self.num_particles, batch_size, self.horizon
            )

            # Aggregate over batch dimension
            rewards = rewards.mean(dim=2)  # [num_samples, num_particles, horizon]
            reward_stds = reward_stds.mean(dim=2)
        else:
            rewards = rewards.view(num_samples, batch_size, self.horizon)
            reward_stds = reward_stds.view(num_samples, batch_size, self.horizon)
            rewards = rewards.mean(dim=1)  # [num_samples, horizon]
            reward_stds = reward_stds.mean(dim=1)

        # Compute discounted returns
        discounts = gamma ** torch.arange(self.horizon, device=device)

        if self.num_particles > 1:
            # Returns for each particle
            returns = torch.sum(
                rewards * discounts.unsqueeze(0).unsqueeze(0),
                dim=-1
            )  # [num_samples, num_particles]
        else:
            returns = torch.sum(rewards * discounts, dim=-1)  # [num_samples]

        # Apply risk metric
        if self.risk_metric == "cvar" and self.num_particles > 1:
            risk_adjusted_returns = self.risk_metrics.cvar(
                returns, alpha=self.risk_level, dim=1
            )
        elif self.risk_metric == "var" and self.num_particles > 1:
            risk_adjusted_returns = self.risk_metrics.var(
                returns, alpha=self.risk_level, dim=1
            )
        elif self.risk_metric == "worst_case" and self.num_particles > 1:
            risk_adjusted_returns = self.risk_metrics.worst_case(
                returns, dim=1
            )
        elif self.risk_metric == "mean_std":
            if self.num_particles > 1:
                mean_returns = returns.mean(dim=1)
                std_returns = returns.std(dim=1)
            else:
                mean_returns = returns
                # Use reward uncertainty for single particle
                std_returns = torch.sum(reward_stds * discounts, dim=-1)

            risk_adjusted_returns = self.risk_metrics.mean_std_penalty(
                mean_returns, std_returns, self.lambda_risk
            )
        elif self.risk_metric == "entropic" and self.num_particles > 1:
            risk_adjusted_returns = self.risk_metrics.entropic_risk(
                returns, beta=self.lambda_risk, dim=1
            )
        else:
            # Default to mean
            if self.num_particles > 1:
                risk_adjusted_returns = returns.mean(dim=1)
            else:
                risk_adjusted_returns = returns

        info = {
            "returns": returns,
            "rewards": rewards,
            "reward_stds": reward_stds,
        }

        if self.num_particles > 1:
            info["return_mean"] = returns.mean(dim=1)
            info["return_std"] = returns.std(dim=1)

        return risk_adjusted_returns, info

    def plan(
        self,
        obs: torch.Tensor,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        return_all_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Plan actions with risk-sensitive optimization.

        Args:
            obs: Current observation
            action_bounds: Action bounds (min, max)
            return_all_info: Return additional planning info

        Returns:
            best_action: Optimal first action
            info: Planning information (if requested)
        """
        device = obs.device
        action_dim = self.world_model.action_dim if hasattr(
            self.world_model, "action_dim"
        ) else obs.shape[-1] - self.world_model.obs_dim

        if self.use_cem:
            # Cross-Entropy Method with risk
            mean = torch.zeros(self.horizon, action_dim, device=device)
            std = torch.ones(self.horizon, action_dim, device=device)

            for iter_idx in range(self.optimization_iters):
                # Sample action sequences
                eps = torch.randn(
                    self.num_samples, self.horizon, action_dim, device=device
                )
                action_sequences = mean + std * eps * self.temperature

                # Clip to bounds
                action_sequences = torch.clamp(
                    action_sequences, action_bounds[0], action_bounds[1]
                )

                # Evaluate with risk
                risk_adjusted_returns, eval_info = self.evaluate_action_sequences(
                    obs, action_sequences
                )

                # Select elite
                elite_idxs = torch.argsort(risk_adjusted_returns, descending=True)[
                    :self.num_elite
                ]
                elite_actions = action_sequences[elite_idxs]

                # Update distribution
                mean = elite_actions.mean(dim=0)
                std = elite_actions.std(dim=0) + 1e-6

            best_action = mean[0]

        else:
            # Random shooting with risk
            action_sequences = torch.rand(
                self.num_samples, self.horizon, action_dim, device=device
            )
            action_sequences = (
                action_sequences * (action_bounds[1] - action_bounds[0])
                + action_bounds[0]
            )

            # Evaluate with risk
            risk_adjusted_returns, eval_info = self.evaluate_action_sequences(
                obs, action_sequences
            )

            # Select best
            best_idx = torch.argmax(risk_adjusted_returns)
            best_action = action_sequences[best_idx, 0]

        if return_all_info:
            info = {
                "risk_adjusted_returns": risk_adjusted_returns,
                "best_return": risk_adjusted_returns.max().item(),
                **eval_info,
            }
            return best_action, info
        else:
            return best_action, None


def compute_trajectory_risk(
    rewards: torch.Tensor,
    uncertainties: Optional[torch.Tensor] = None,
    risk_metric: str = "cvar",
    risk_level: float = 0.1,
    lambda_risk: float = 1.0,
) -> torch.Tensor:
    """
    Compute risk metrics for trajectory rewards.

    Args:
        rewards: Trajectory rewards [batch_size, num_particles, horizon]
                 or [batch_size, horizon]
        uncertainties: Optional uncertainty estimates
        risk_metric: Type of risk metric
        risk_level: Risk level for CVaR/VaR
        lambda_risk: Risk aversion parameter

    Returns:
        Risk-adjusted values
    """
    metrics = RiskMetrics()

    # Check if we have particle dimension
    has_particles = len(rewards.shape) == 3

    if has_particles:
        # Compute returns
        returns = rewards.sum(dim=-1)  # Sum over horizon

        if risk_metric == "cvar":
            return metrics.cvar(returns, alpha=risk_level, dim=1)
        elif risk_metric == "var":
            return metrics.var(returns, alpha=risk_level, dim=1)
        elif risk_metric == "worst_case":
            return metrics.worst_case(returns, dim=1)
        elif risk_metric == "mean_std":
            mean_returns = returns.mean(dim=1)
            std_returns = returns.std(dim=1)
            return metrics.mean_std_penalty(mean_returns, std_returns, lambda_risk)
        elif risk_metric == "entropic":
            return metrics.entropic_risk(returns, beta=lambda_risk, dim=1)
        else:
            return returns.mean(dim=1)
    else:
        # No particles, use provided uncertainties if available
        returns = rewards.sum(dim=-1)

        if risk_metric == "mean_std" and uncertainties is not None:
            uncertainty_sum = uncertainties.sum(dim=-1)
            return metrics.mean_std_penalty(returns, uncertainty_sum, lambda_risk)
        else:
            return returns