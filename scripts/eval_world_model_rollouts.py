"""
Long-horizon evaluation of world model predictions vs reality.

Systematically evaluates world model accuracy over extended rollouts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

from envs import Simple3DNavEnv
from models import WorldModel, EnsembleWorldModel
import config


class WorldModelEvaluator:
    """Evaluates world model prediction accuracy over long horizons."""

    def __init__(
        self,
        env: Simple3DNavEnv,
        world_model: nn.Module,
        device: str = "cpu"
    ):
        """
        Initialize evaluator.

        Args:
            env: Environment for ground truth rollouts
            world_model: Trained world model
            device: Device for model inference
        """
        self.env = env
        self.world_model = world_model.to(device)
        self.world_model.eval()
        self.device = device

    def rollout_comparison(
        self,
        initial_obs: np.ndarray,
        actions: np.ndarray,
    ) -> Dict:
        """
        Compare world model rollout with ground truth.

        Args:
            initial_obs: Initial observation
            actions: Sequence of actions to execute [horizon, action_dim]

        Returns:
            Dictionary of metrics and trajectories
        """
        horizon = len(actions)
        is_ensemble = isinstance(self.world_model, EnsembleWorldModel)

        # Ground truth rollout
        true_obs = [initial_obs]
        true_rewards = []

        # Reset environment to initial state
        self.env.state = initial_obs[:6].copy()
        self.env.goal = initial_obs[6:9].copy()

        for action in actions:
            obs, reward, done, info = self.env.step(action)
            true_obs.append(obs)
            true_rewards.append(reward)

        true_obs = np.stack(true_obs)
        true_rewards = np.array(true_rewards)

        # Model rollout
        with torch.no_grad():
            model_obs = [initial_obs]
            model_rewards = []

            # For ensemble, track uncertainty
            if is_ensemble:
                model_obs_std = []
                model_rewards_std = []

            obs_tensor = torch.from_numpy(initial_obs).float().to(self.device)

            for action in actions:
                action_tensor = torch.from_numpy(action).float().to(self.device)

                if is_ensemble:
                    # Get mean and std from ensemble
                    (next_obs_mean, next_obs_std), (reward_mean, reward_std) = self.world_model(
                        obs_tensor.unsqueeze(0),
                        action_tensor.unsqueeze(0),
                        reduce="mean_std"
                    )

                    next_obs_pred = next_obs_mean.squeeze(0)
                    reward_pred = reward_mean.item()

                    model_obs_std.append(next_obs_std.squeeze(0).cpu().numpy())
                    model_rewards_std.append(reward_std.item())
                else:
                    # Single model prediction
                    next_obs_pred, reward_pred = self.world_model(
                        obs_tensor.unsqueeze(0),
                        action_tensor.unsqueeze(0)
                    )

                    next_obs_pred = next_obs_pred.squeeze(0)
                    reward_pred = reward_pred.item()

                model_obs.append(next_obs_pred.cpu().numpy())
                model_rewards.append(reward_pred)

                # Update for next step
                obs_tensor = next_obs_pred

        model_obs = np.stack(model_obs)
        model_rewards = np.array(model_rewards)

        # Compute metrics
        obs_errors = np.linalg.norm(model_obs[1:] - true_obs[1:], axis=-1)
        reward_errors = np.abs(model_rewards - true_rewards)

        # Position errors (first 3 dims)
        position_errors = np.linalg.norm(
            model_obs[1:, :3] - true_obs[1:, :3],
            axis=-1
        )

        # Velocity errors (dims 3-6)
        velocity_errors = np.linalg.norm(
            model_obs[1:, 3:6] - true_obs[1:, 3:6],
            axis=-1
        )

        # Goal reaching analysis
        true_final_dist = np.linalg.norm(true_obs[-1, :3] - true_obs[-1, 6:9])
        model_final_dist = np.linalg.norm(model_obs[-1, :3] - model_obs[-1, 6:9])

        true_success = true_final_dist < self.env.goal_radius
        model_success = model_final_dist < self.env.goal_radius

        result = {
            "true_obs": true_obs,
            "model_obs": model_obs,
            "true_rewards": true_rewards,
            "model_rewards": model_rewards,
            "obs_errors": obs_errors,
            "reward_errors": reward_errors,
            "position_errors": position_errors,
            "velocity_errors": velocity_errors,
            "true_final_dist": true_final_dist,
            "model_final_dist": model_final_dist,
            "success_match": true_success == model_success,
            "horizon": horizon,
        }

        # Add ensemble uncertainty metrics if available
        if is_ensemble:
            model_obs_std = np.stack(model_obs_std)
            model_rewards_std = np.array(model_rewards_std)

            result["model_obs_std"] = model_obs_std
            result["model_rewards_std"] = model_rewards_std

            # Compute uncertainty correlation with error
            uncertainty_error_corr = np.corrcoef(
                model_obs_std.mean(axis=-1),
                obs_errors
            )[0, 1]

            result["uncertainty_error_correlation"] = uncertainty_error_corr

        return result

    def evaluate_episodes(
        self,
        num_episodes: int = 100,
        horizon: int = 50,
        policy: str = "random",
    ) -> Dict:
        """
        Evaluate world model over multiple episodes.

        Args:
            num_episodes: Number of test episodes
            horizon: Rollout horizon
            policy: Policy for action selection

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        for episode in tqdm(range(num_episodes), desc="Evaluating episodes"):
            # Reset environment
            initial_obs = self.env.reset()

            # Generate action sequence
            if policy == "random":
                actions = self.env.rng.uniform(
                    -self.env.max_acceleration,
                    self.env.max_acceleration,
                    size=(horizon, 3)
                )
            elif policy == "heuristic":
                # Simple goal-directed actions
                actions = []
                obs = initial_obs
                for _ in range(horizon):
                    position = obs[:3]
                    velocity = obs[3:6]
                    goal = obs[6:9]

                    direction = goal - position
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction = direction / distance

                    desired_velocity = direction * min(distance * 0.5, self.env.max_velocity)
                    acceleration = (desired_velocity - velocity) * 2.0
                    acceleration = np.clip(
                        acceleration,
                        -self.env.max_acceleration,
                        self.env.max_acceleration
                    )
                    actions.append(acceleration)

                    # Simulate for next action (this is cheating but ok for eval)
                    obs, _, _, _ = self.env.step(acceleration)

                actions = np.stack(actions)

                # Reset env for fair comparison
                self.env.state = initial_obs[:6].copy()
                self.env.goal = initial_obs[6:9].copy()
            else:
                raise ValueError(f"Unknown policy: {policy}")

            # Run comparison
            metrics = self.rollout_comparison(initial_obs, actions)
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        return aggregated

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across episodes."""
        horizon = metrics_list[0]["horizon"]

        # Stack errors across episodes
        obs_errors = np.stack([m["obs_errors"] for m in metrics_list])
        reward_errors = np.stack([m["reward_errors"] for m in metrics_list])
        position_errors = np.stack([m["position_errors"] for m in metrics_list])
        velocity_errors = np.stack([m["velocity_errors"] for m in metrics_list])

        # Compute statistics per timestep
        mean_obs_error = obs_errors.mean(axis=0)
        std_obs_error = obs_errors.std(axis=0)
        mean_reward_error = reward_errors.mean(axis=0)
        std_reward_error = reward_errors.std(axis=0)
        mean_position_error = position_errors.mean(axis=0)
        std_position_error = position_errors.std(axis=0)
        mean_velocity_error = velocity_errors.mean(axis=0)
        std_velocity_error = velocity_errors.std(axis=0)

        # Final distance metrics
        true_final_dists = [m["true_final_dist"] for m in metrics_list]
        model_final_dists = [m["model_final_dist"] for m in metrics_list]
        success_matches = [m["success_match"] for m in metrics_list]

        aggregated = {
            "horizon": horizon,
            "num_episodes": len(metrics_list),
            "mean_obs_error": mean_obs_error,
            "std_obs_error": std_obs_error,
            "mean_reward_error": mean_reward_error,
            "std_reward_error": std_reward_error,
            "mean_position_error": mean_position_error,
            "std_position_error": std_position_error,
            "mean_velocity_error": mean_velocity_error,
            "std_velocity_error": std_velocity_error,
            "mean_final_obs_error": mean_obs_error[-1],
            "mean_final_position_error": mean_position_error[-1],
            "success_match_rate": np.mean(success_matches),
            "true_final_dist_mean": np.mean(true_final_dists),
            "model_final_dist_mean": np.mean(model_final_dists),
        }

        # Aggregate ensemble uncertainty metrics if available
        if "model_obs_std" in metrics_list[0]:
            obs_uncertainties = np.stack([m["model_obs_std"] for m in metrics_list])
            reward_uncertainties = np.stack([m["model_rewards_std"] for m in metrics_list])

            mean_obs_uncertainty = obs_uncertainties.mean(axis=0).mean(axis=-1)
            mean_reward_uncertainty = reward_uncertainties.mean(axis=0)

            # Uncertainty-error correlations
            uncertainty_corrs = [m["uncertainty_error_correlation"] for m in metrics_list
                                if not np.isnan(m["uncertainty_error_correlation"])]

            aggregated["mean_obs_uncertainty"] = mean_obs_uncertainty
            aggregated["mean_reward_uncertainty"] = mean_reward_uncertainty
            aggregated["mean_uncertainty_error_correlation"] = np.mean(uncertainty_corrs) if uncertainty_corrs else 0.0
            aggregated["is_ensemble"] = True
        else:
            aggregated["is_ensemble"] = False

        return aggregated

    def plot_results(self, metrics: Dict, save_dir: Path):
        """Generate evaluation plots."""
        save_dir.mkdir(exist_ok=True, parents=True)
        horizon = metrics["horizon"]
        timesteps = np.arange(1, horizon + 1)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Observation error over time
        ax = axes[0, 0]
        ax.plot(timesteps, metrics["mean_obs_error"], 'b-', label='Mean')
        ax.fill_between(
            timesteps,
            metrics["mean_obs_error"] - metrics["std_obs_error"],
            metrics["mean_obs_error"] + metrics["std_obs_error"],
            alpha=0.3
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Observation Error (L2)")
        ax.set_title("Observation Prediction Error vs Horizon")
        ax.grid(True)
        ax.legend()

        # Plot 2: Position error over time
        ax = axes[0, 1]
        ax.plot(timesteps, metrics["mean_position_error"], 'g-', label='Mean')
        ax.fill_between(
            timesteps,
            metrics["mean_position_error"] - metrics["std_position_error"],
            metrics["mean_position_error"] + metrics["std_position_error"],
            alpha=0.3
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Position Error (L2)")
        ax.set_title("Position Prediction Error vs Horizon")
        ax.grid(True)
        ax.legend()

        # Plot 3: Velocity error over time
        ax = axes[0, 2]
        ax.plot(timesteps, metrics["mean_velocity_error"], 'r-', label='Mean')
        ax.fill_between(
            timesteps,
            metrics["mean_velocity_error"] - metrics["std_velocity_error"],
            metrics["mean_velocity_error"] + metrics["std_velocity_error"],
            alpha=0.3
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Velocity Error (L2)")
        ax.set_title("Velocity Prediction Error vs Horizon")
        ax.grid(True)
        ax.legend()

        # Plot 4: Reward error over time
        ax = axes[1, 0]
        ax.plot(timesteps, metrics["mean_reward_error"], 'orange', label='Mean')
        ax.fill_between(
            timesteps,
            metrics["mean_reward_error"] - metrics["std_reward_error"],
            metrics["mean_reward_error"] + metrics["std_reward_error"],
            alpha=0.3
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward Error (Abs)")
        ax.set_title("Reward Prediction Error vs Horizon")
        ax.grid(True)
        ax.legend()

        # Plot 5: Error growth rate OR Uncertainty vs Error
        ax = axes[1, 1]
        if metrics.get("is_ensemble", False):
            # Plot uncertainty vs error correlation for ensemble
            ax.plot(timesteps, metrics["mean_obs_uncertainty"], 'purple', label='Uncertainty')
            ax2 = ax.twinx()
            ax2.plot(timesteps, metrics["mean_obs_error"], 'orange', label='Error')
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Model Uncertainty", color='purple')
            ax2.set_ylabel("Prediction Error", color='orange')
            ax.set_title(f"Uncertainty vs Error (ρ={metrics.get('mean_uncertainty_error_correlation', 0):.2f})")
            ax.grid(True)
        else:
            # Regular error growth plot
            error_growth = metrics["mean_obs_error"] / metrics["mean_obs_error"][0]
            ax.plot(timesteps, error_growth, 'purple')
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Relative Error Growth")
            ax.set_title("Error Growth Factor (relative to t=1)")
            ax.grid(True)

        # Plot 6: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
Summary Statistics:
==================
Episodes: {metrics['num_episodes']}
Horizon: {metrics['horizon']}

Final Errors:
- Observation: {metrics['mean_final_obs_error']:.4f}
- Position: {metrics['mean_final_position_error']:.4f}

Success Match Rate: {metrics['success_match_rate']:.1%}

Goal Distance:
- True: {metrics['true_final_dist_mean']:.3f}
- Model: {metrics['model_final_dist_mean']:.3f}
"""
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle("World Model Long-Horizon Evaluation", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_path = save_dir / "world_model_evaluation.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved evaluation plot to {plot_path}")

        # Additional plot: Error distribution at different horizons
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        horizons_to_plot = [10, 25, horizon-1]
        for idx, h in enumerate(horizons_to_plot):
            ax = axes[idx]
            ax.hist(metrics.get(f"obs_errors_t{h}", []), bins=30, alpha=0.7)
            ax.set_xlabel("Observation Error")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Error Distribution at t={h+1}")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Error Distributions at Different Horizons", fontsize=14)
        plt.tight_layout()

        dist_plot_path = save_dir / "error_distributions.png"
        plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved distribution plot to {dist_plot_path}")


def main():
    """Main evaluation routine."""
    parser = argparse.ArgumentParser(description="Evaluate world model rollouts")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.MODEL_PATHS["best_model"]),
        help="Path to trained model"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Rollout horizon"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "heuristic"],
        default="random",
        help="Policy for action selection"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE_CONFIG["device"],
        help="Device for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.LOGS_DIR / "evaluation"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        default=config.MODEL_CONFIG.get("use_ensemble", False),
        help="Use ensemble world model"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=config.MODEL_CONFIG.get("ensemble_size", 5),
        help="Number of ensemble members"
    )

    args = parser.parse_args()

    # Create environment
    env = Simple3DNavEnv(**config.ENV_CONFIG)

    # Get dimensions
    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]

    # Create and load world model
    if args.use_ensemble:
        # Load ensemble model
        print(f"Loading ensemble model with {args.ensemble_size} members...")

        # Create ensemble model
        world_model = EnsembleWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=args.ensemble_size,
            **config.MODEL_CONFIG["world_model"],
        )

        # Load each member
        members_loaded = 0
        for member_idx in range(args.ensemble_size):
            member_path = config.WEIGHTS_DIR / f"world_model_member_{member_idx}.pt"
            if member_path.exists():
                checkpoint = torch.load(member_path, map_location=args.device)
                world_model.models[member_idx].load_state_dict(checkpoint["model_state_dict"])
                members_loaded += 1
                print(f"  Loaded member {member_idx} from {member_path}")
            else:
                print(f"  Warning: Member {member_idx} not found at {member_path}")

        if members_loaded == 0:
            print("Error: No ensemble members found!")
            print("Please train the ensemble model first: python training/train_world_model.py --use_ensemble")
            return

        if members_loaded < args.ensemble_size:
            print(f"Warning: Only {members_loaded}/{args.ensemble_size} members loaded")
    else:
        # Load single model
        if not Path(args.model_path).exists():
            print(f"Error: Model not found at {args.model_path}")
            print("Please train the model first: python training/train_world_model.py")
            return

        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=args.device)

        # Create single world model
        world_model = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **config.MODEL_CONFIG["world_model"],
        )
        world_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Create evaluator
    evaluator = WorldModelEvaluator(env, world_model, device=args.device)

    # Run evaluation
    print(f"\nEvaluating {args.num_episodes} episodes with horizon {args.horizon}...")
    print(f"Policy: {args.policy}")

    metrics = evaluator.evaluate_episodes(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        policy=args.policy
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            else:
                json_metrics[key] = value
        json.dump(json_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Generate plots
    evaluator.plot_results(metrics, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("World Model Evaluation Summary")
    print("="*60)
    print(f"Episodes evaluated: {metrics['num_episodes']}")
    print(f"Horizon: {metrics['horizon']}")
    print(f"Policy: {args.policy}")
    print(f"Model type: {'Ensemble' if args.use_ensemble else 'Single'}")
    print("\nKey Metrics:")
    print(f"  Mean final observation error: {metrics['mean_final_obs_error']:.4f}")
    print(f"  Mean final position error: {metrics['mean_final_position_error']:.4f}")
    print(f"  Success match rate: {metrics['success_match_rate']:.1%}")
    print(f"  True final goal distance: {metrics['true_final_dist_mean']:.3f}")
    print(f"  Model final goal distance: {metrics['model_final_dist_mean']:.3f}")

    if metrics.get("is_ensemble", False):
        print("\nEnsemble Metrics:")
        print(f"  Mean uncertainty-error correlation: {metrics['mean_uncertainty_error_correlation']:.3f}")
        print(f"  Final observation uncertainty: {metrics['mean_obs_uncertainty'][-1]:.4f}")
        if metrics['mean_uncertainty_error_correlation'] > 0.5:
            print("  ✓ Good uncertainty calibration (ρ > 0.5)")

    # Determine model quality
    if metrics['mean_final_obs_error'] < 0.1:
        quality = "Excellent"
    elif metrics['mean_final_obs_error'] < 0.5:
        quality = "Good"
    elif metrics['mean_final_obs_error'] < 1.0:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"\nModel Quality Assessment: {quality}")
    print(f"The model maintains predictions for ~{np.where(metrics['mean_obs_error'] > 0.5)[0][0] if np.any(metrics['mean_obs_error'] > 0.5) else metrics['horizon']} steps before significant drift")


if __name__ == "__main__":
    # Add missing import
    import torch.nn as nn
    main()