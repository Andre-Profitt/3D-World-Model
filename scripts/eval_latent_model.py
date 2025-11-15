"""
Evaluate latent world model rollout accuracy.

Compares latent model predictions against ground truth in both
latent and observation spaces.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from envs import Simple3DNavEnv
from models import (
    Encoder,
    Decoder,
    LatentWorldModel,
    LatentMPCWrapper
)
import config


class LatentModelEvaluator:
    """Evaluates latent world model predictions."""

    def __init__(
        self,
        env: Simple3DNavEnv,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_model: nn.Module,
        device: str = "cpu"
    ):
        """
        Initialize evaluator.

        Args:
            env: Environment for ground truth rollouts
            encoder: Trained encoder
            decoder: Trained decoder
            latent_model: Trained latent dynamics model
            device: Device for inference
        """
        self.env = env
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.latent_model = latent_model.to(device)
        self.device = device

        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()
        self.latent_model.eval()

        # Create wrapper for convenience
        self.wrapper = LatentMPCWrapper(
            encoder=self.encoder,
            decoder=self.decoder,
            latent_world_model=self.latent_model,
            device=torch.device(device)
        )

    def rollout_comparison(
        self,
        initial_obs: np.ndarray,
        actions: np.ndarray,
    ) -> Dict:
        """
        Compare latent model rollout with ground truth.

        Args:
            initial_obs: Initial observation
            actions: Action sequence [horizon, action_dim]

        Returns:
            Dictionary of metrics
        """
        horizon = len(actions)

        # Ground truth rollout
        true_obs = [initial_obs]
        true_rewards = []

        # Reset environment
        self.env.state = initial_obs[:6].copy()
        self.env.goal = initial_obs[6:9].copy()

        for action in actions:
            obs, reward, done, info = self.env.step(action)
            true_obs.append(obs)
            true_rewards.append(reward)

        true_obs = np.stack(true_obs)
        true_rewards = np.array(true_rewards)

        # Convert to tensors
        initial_obs_t = torch.from_numpy(initial_obs).float().to(self.device)
        actions_t = torch.from_numpy(actions).float().to(self.device)

        # Model rollout in latent space
        with torch.no_grad():
            # Encode initial observation
            initial_latent = self.wrapper.encode(initial_obs_t.unsqueeze(0))

            # Track predictions
            model_latents = [initial_latent]
            model_obs = [initial_obs]
            model_rewards = []

            # Also track latent space errors
            true_latents = [initial_latent]  # First latent is same

            current_latent = initial_latent

            for t, action in enumerate(actions):
                action_t = torch.from_numpy(action).float().to(self.device).unsqueeze(0)

                # Predict next latent and reward
                next_latent, reward = self.latent_model(current_latent, action_t)

                # Decode to observation
                next_obs = self.wrapper.decode(next_latent)

                model_latents.append(next_latent)
                model_obs.append(next_obs.squeeze(0).cpu().numpy())
                model_rewards.append(reward.item())

                # Encode true next observation for latent comparison
                true_next_obs = torch.from_numpy(true_obs[t+1]).float().to(self.device).unsqueeze(0)
                true_next_latent = self.wrapper.encode(true_next_obs)
                true_latents.append(true_next_latent)

                current_latent = next_latent

        model_obs = np.stack(model_obs)
        model_rewards = np.array(model_rewards)

        # Stack latents
        model_latents = torch.cat(model_latents, dim=0).cpu().numpy()
        true_latents = torch.cat(true_latents, dim=0).cpu().numpy()

        # Compute errors
        obs_errors = np.linalg.norm(model_obs[1:] - true_obs[1:], axis=-1)
        reward_errors = np.abs(model_rewards - true_rewards)
        latent_errors = np.linalg.norm(model_latents[1:] - true_latents[1:], axis=-1)

        # Position and velocity errors
        position_errors = np.linalg.norm(
            model_obs[1:, :3] - true_obs[1:, :3],
            axis=-1
        )
        velocity_errors = np.linalg.norm(
            model_obs[1:, 3:6] - true_obs[1:, 3:6],
            axis=-1
        )

        # Reconstruction errors (encode true obs, decode, compare)
        with torch.no_grad():
            true_obs_t = torch.from_numpy(true_obs).float().to(self.device)
            encoded = self.wrapper.encode(true_obs_t)
            reconstructed = self.wrapper.decode(encoded).cpu().numpy()
            reconstruction_errors = np.linalg.norm(reconstructed - true_obs, axis=-1)

        result = {
            "true_obs": true_obs,
            "model_obs": model_obs,
            "true_rewards": true_rewards,
            "model_rewards": model_rewards,
            "model_latents": model_latents,
            "true_latents": true_latents,
            "obs_errors": obs_errors,
            "reward_errors": reward_errors,
            "latent_errors": latent_errors,
            "position_errors": position_errors,
            "velocity_errors": velocity_errors,
            "reconstruction_errors": reconstruction_errors,
            "horizon": horizon,
        }

        return result

    def evaluate_episodes(
        self,
        num_episodes: int = 100,
        horizon: int = 50,
        policy: str = "random",
    ) -> Dict:
        """
        Evaluate over multiple episodes.

        Args:
            num_episodes: Number of evaluation episodes
            horizon: Rollout horizon
            policy: Action selection policy

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            # Reset environment
            initial_obs = self.env.reset()

            # Generate actions
            if policy == "random":
                actions = self.env.rng.uniform(
                    -self.env.max_acceleration,
                    self.env.max_acceleration,
                    size=(horizon, 3)
                )
            else:
                raise ValueError(f"Unknown policy: {policy}")

            # Run comparison
            metrics = self.rollout_comparison(initial_obs, actions)
            all_metrics.append(metrics)

        # Aggregate
        aggregated = self._aggregate_metrics(all_metrics)
        return aggregated

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across episodes."""
        horizon = metrics_list[0]["horizon"]

        # Stack errors
        obs_errors = np.stack([m["obs_errors"] for m in metrics_list])
        reward_errors = np.stack([m["reward_errors"] for m in metrics_list])
        latent_errors = np.stack([m["latent_errors"] for m in metrics_list])
        position_errors = np.stack([m["position_errors"] for m in metrics_list])
        velocity_errors = np.stack([m["velocity_errors"] for m in metrics_list])
        reconstruction_errors = np.stack([m["reconstruction_errors"] for m in metrics_list])

        # Compute statistics
        aggregated = {
            "horizon": horizon,
            "num_episodes": len(metrics_list),
            "mean_obs_error": obs_errors.mean(axis=0),
            "std_obs_error": obs_errors.std(axis=0),
            "mean_reward_error": reward_errors.mean(axis=0),
            "std_reward_error": reward_errors.std(axis=0),
            "mean_latent_error": latent_errors.mean(axis=0),
            "std_latent_error": latent_errors.std(axis=0),
            "mean_position_error": position_errors.mean(axis=0),
            "std_position_error": position_errors.std(axis=0),
            "mean_velocity_error": velocity_errors.mean(axis=0),
            "std_velocity_error": velocity_errors.std(axis=0),
            "mean_reconstruction_error": reconstruction_errors.mean(axis=0),
            "std_reconstruction_error": reconstruction_errors.std(axis=0),
            "mean_final_obs_error": obs_errors[:, -1].mean(),
            "mean_final_latent_error": latent_errors[:, -1].mean(),
            "mean_final_position_error": position_errors[:, -1].mean(),
        }

        # Compute effective horizon (where error stays below threshold)
        threshold = 0.5  # Observation error threshold
        effective_horizons = []
        for episode_errors in obs_errors:
            above_threshold = np.where(episode_errors > threshold)[0]
            if len(above_threshold) > 0:
                effective_horizons.append(above_threshold[0])
            else:
                effective_horizons.append(horizon)

        aggregated["mean_effective_horizon"] = np.mean(effective_horizons)
        aggregated["std_effective_horizon"] = np.std(effective_horizons)

        return aggregated

    def plot_results(self, metrics: Dict, save_dir: Path):
        """Generate evaluation plots."""
        save_dir.mkdir(exist_ok=True, parents=True)
        horizon = metrics["horizon"]
        timesteps = np.arange(1, horizon + 1)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot 1: Observation error
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
        ax.set_title("Observation Space Prediction Error")
        ax.grid(True)
        ax.legend()

        # Plot 2: Latent error
        ax = axes[0, 1]
        ax.plot(timesteps, metrics["mean_latent_error"], 'g-', label='Mean')
        ax.fill_between(
            timesteps,
            metrics["mean_latent_error"] - metrics["std_latent_error"],
            metrics["mean_latent_error"] + metrics["std_latent_error"],
            alpha=0.3
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Latent Error (L2)")
        ax.set_title("Latent Space Prediction Error")
        ax.grid(True)
        ax.legend()

        # Plot 3: Position error
        ax = axes[0, 2]
        ax.plot(timesteps, metrics["mean_position_error"], 'r-', label='Mean')
        ax.fill_between(
            timesteps,
            metrics["mean_position_error"] - metrics["std_position_error"],
            metrics["mean_position_error"] + metrics["std_position_error"],
            alpha=0.3
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Position Error (L2)")
        ax.set_title("Position Prediction Error")
        ax.grid(True)
        ax.legend()

        # Plot 4: Reconstruction error
        ax = axes[1, 0]
        ax.plot(np.arange(horizon + 1), metrics["mean_reconstruction_error"], 'purple', label='Mean')
        ax.fill_between(
            np.arange(horizon + 1),
            metrics["mean_reconstruction_error"] - metrics["std_reconstruction_error"],
            metrics["mean_reconstruction_error"] + metrics["std_reconstruction_error"],
            alpha=0.3
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reconstruction Error")
        ax.set_title("Autoencoder Reconstruction Quality")
        ax.grid(True)
        ax.legend()

        # Plot 5: Comparison of error growth
        ax = axes[1, 1]
        obs_growth = metrics["mean_obs_error"] / metrics["mean_obs_error"][0]
        latent_growth = metrics["mean_latent_error"] / metrics["mean_latent_error"][0]
        ax.plot(timesteps, obs_growth, 'b-', label='Observation Space')
        ax.plot(timesteps, latent_growth, 'g-', label='Latent Space')
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Relative Error Growth")
        ax.set_title("Error Growth Comparison")
        ax.grid(True)
        ax.legend()

        # Plot 6: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
Latent Model Evaluation Summary
===============================
Episodes: {metrics['num_episodes']}
Horizon: {metrics['horizon']}

Dimensions:
- Observation: 9D
- Latent: 16D

Final Errors:
- Observation: {metrics['mean_final_obs_error']:.4f}
- Latent: {metrics['mean_final_latent_error']:.4f}
- Position: {metrics['mean_final_position_error']:.4f}

Effective Horizon: {metrics['mean_effective_horizon']:.1f} ± {metrics['std_effective_horizon']:.1f}
(steps before error > 0.5)

Reconstruction Error: {metrics['mean_reconstruction_error'][0]:.4f}
"""
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle("Latent World Model Evaluation", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_path = save_dir / "latent_model_evaluation.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved evaluation plot to {plot_path}")

        return plot_path


def main():
    """Main evaluation routine."""
    parser = argparse.ArgumentParser(description="Evaluate latent world model")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Rollout horizon"
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
        default=str(config.LOGS_DIR / "latent_evaluation"),
        help="Output directory"
    )

    args = parser.parse_args()

    # Create environment
    env = Simple3DNavEnv(**config.ENV_CONFIG)
    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]

    # Load models
    print("Loading latent world model components...")

    # Load encoder
    latent_dim = config.MODEL_CONFIG["encoder"]["latent_dim"]
    encoder = Encoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        **{k: v for k, v in config.MODEL_CONFIG["encoder"].items() if k != "latent_dim"}
    )
    encoder_checkpoint = torch.load(
        config.WEIGHTS_DIR / "encoder.pt",
        map_location=args.device
    )
    encoder.load_state_dict(encoder_checkpoint["model_state_dict"])
    print(f"Loaded encoder: {obs_dim}D → {latent_dim}D")

    # Load decoder
    decoder = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        **{k: v for k, v in config.MODEL_CONFIG["decoder"].items() if k != "latent_dim"}
    )
    decoder_checkpoint = torch.load(
        config.WEIGHTS_DIR / "decoder.pt",
        map_location=args.device
    )
    decoder.load_state_dict(decoder_checkpoint["model_state_dict"])
    print(f"Loaded decoder: {latent_dim}D → {obs_dim}D")

    # Load latent dynamics model
    # Filter out training-specific parameters
    latent_config = {
        k: v for k, v in config.MODEL_CONFIG["latent_world_model"].items()
        if k not in ["beta_recon", "lr", "weight_decay"]
    }
    latent_model = LatentWorldModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        action_dim=action_dim,
        **latent_config
    )

    # Load only dynamics and reward networks
    latent_checkpoint = torch.load(
        config.WEIGHTS_DIR / "best_latent_world_model.pt",
        map_location=args.device
    )

    # Extract dynamics and reward network states
    dynamics_state = {}
    reward_state = {}
    for key, value in latent_checkpoint["model_state_dict"].items():
        if key.startswith("dynamics_net."):
            dynamics_state[key.replace("dynamics_net.", "")] = value
        elif key.startswith("reward_net."):
            reward_state[key.replace("reward_net.", "")] = value

    latent_model.dynamics_net.load_state_dict(dynamics_state)
    latent_model.reward_net.load_state_dict(reward_state)
    print(f"Loaded latent dynamics model")

    # Create evaluator
    evaluator = LatentModelEvaluator(
        env=env,
        encoder=encoder,
        decoder=decoder,
        latent_model=latent_model,
        device=args.device
    )

    # Run evaluation
    print(f"\nEvaluating {args.num_episodes} episodes with horizon {args.horizon}...")
    metrics = evaluator.evaluate_episodes(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        policy="random"
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics
    metrics_path = output_dir / "latent_metrics.json"
    with open(metrics_path, "w") as f:
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                json_metrics[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                json_metrics[key] = int(value)
            else:
                json_metrics[key] = value
        json.dump(json_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Generate plots
    plot_path = evaluator.plot_results(metrics, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("Latent World Model Evaluation Results")
    print("="*60)
    print(f"Architecture: 9D obs → 16D latent → dynamics → 16D latent → 9D obs")
    print(f"Episodes evaluated: {metrics['num_episodes']}")
    print(f"Horizon: {metrics['horizon']}")
    print("\nKey Metrics:")
    print(f"  Final observation error: {metrics['mean_final_obs_error']:.4f}")
    print(f"  Final latent error: {metrics['mean_final_latent_error']:.4f}")
    print(f"  Final position error: {metrics['mean_final_position_error']:.4f}")
    print(f"  Effective horizon: {metrics['mean_effective_horizon']:.1f} ± {metrics['std_effective_horizon']:.1f} steps")
    print(f"  Reconstruction quality: {metrics['mean_reconstruction_error'][0]:.4f}")

    # Model quality assessment
    if metrics['mean_final_obs_error'] < 0.1:
        quality = "Excellent"
    elif metrics['mean_final_obs_error'] < 0.5:
        quality = "Good"
    elif metrics['mean_final_obs_error'] < 1.0:
        quality = "Fair"
    else:
        quality = "Needs improvement"

    print(f"\nModel Quality: {quality}")
    print(f"The latent model maintains accurate predictions for ~{int(metrics['mean_effective_horizon'])} steps")

    if metrics['mean_reconstruction_error'][0] < 0.02:
        print("✓ Excellent autoencoder quality (error < 0.02)")

    print(f"\nVisualization saved to: {plot_path}")


if __name__ == "__main__":
    main()