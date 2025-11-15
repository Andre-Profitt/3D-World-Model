#!/usr/bin/env python3
"""
Comprehensive rollout evaluation for VAE + Latent Dynamics system.
Generates detailed error vs horizon plots and trajectory comparisons.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Tuple, List, Dict
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VAE
from models.simple_latent_dynamics import SimpleLatentDynamics
from envs.simple_3d_nav import Simple3DNavEnv


def load_normalization_stats(path='weights/normalization_stats.json'):
    """Load normalization statistics."""
    with open(path, 'r') as f:
        stats = json.load(f)
    return np.array(stats['mean']), np.array(stats['std'])


def normalize(obs, mean, std):
    """Normalize observation."""
    return (obs - mean) / (std + 1e-8)


def denormalize(obs, mean, std):
    """Denormalize observation."""
    return obs * std + mean


def load_models(vae_path: str, dynamics_path: str):
    """Load VAE and dynamics models."""
    # Load VAE
    vae_checkpoint = torch.load(vae_path, map_location='cpu', weights_only=True)
    vae_config = vae_checkpoint.get('config', {})

    vae = VAE(
        obs_dim=vae_config.get('obs_dim', 9),
        latent_dim=vae_config.get('latent_dim', 24),
        encoder_hidden=vae_config.get('encoder_hidden', [256, 256, 256]),
        decoder_hidden=vae_config.get('decoder_hidden', [256, 256, 256]),
        activation=vae_config.get('activation', 'elu'),
        beta=vae_config.get('beta', 0.001),
    )
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()

    # Load dynamics
    dynamics_checkpoint = torch.load(dynamics_path, map_location='cpu', weights_only=True)

    dynamics = SimpleLatentDynamics(
        latent_dim=dynamics_checkpoint['latent_dim'],
        action_dim=dynamics_checkpoint['action_dim'],
        hidden_dim=dynamics_checkpoint['hidden_dim'],
        n_layers=dynamics_checkpoint['n_layers'],
    )
    dynamics.load_state_dict(dynamics_checkpoint['model_state_dict'])
    dynamics.eval()

    return vae, dynamics


def evaluate_rollout_quality(vae, dynamics, env, mean, std,
                            horizon=50, n_episodes=100) -> Dict:
    """Evaluate rollout quality at different horizons."""

    print(f"\nEvaluating rollout quality over {n_episodes} episodes...")

    # Store errors for different metrics
    results = {
        'position_errors': [],  # L2 error in position (first 3 dims)
        'velocity_errors': [],  # L2 error in velocity (dims 3-6)
        'goal_errors': [],      # L2 error in goal position (last 3 dims)
        'total_errors': [],     # Total L2 error
        'normalized_errors': [] # Error in normalized space
    }

    for episode in tqdm(range(n_episodes), desc="Rollout evaluation"):
        # Reset environment
        true_obs = env.reset()

        # Generate random action sequence
        actions = []
        for _ in range(horizon):
            action = np.random.uniform(-1, 1, size=env.action_dim)
            actions.append(action)
        actions = torch.FloatTensor(np.array(actions))

        # Normalize and encode initial state
        obs_normalized = normalize(true_obs, mean, std)
        with torch.no_grad():
            z0 = vae.encode(torch.FloatTensor(obs_normalized).unsqueeze(0))

        # Store errors for this episode
        episode_results = {key: [] for key in results.keys()}

        # Rollout
        z = z0
        for t in range(horizon):
            # Predict next latent
            with torch.no_grad():
                z_next, _ = dynamics(z, actions[t].unsqueeze(0))
                z = z_next

            # Decode to normalized observation
            with torch.no_grad():
                pred_obs_normalized = vae.decode(z).squeeze(0).numpy()

            # Denormalize for comparison
            pred_obs = denormalize(pred_obs_normalized, mean, std)

            # True next state
            true_obs, _, done, _ = env.step(actions[t].numpy())
            true_obs_normalized = normalize(true_obs, mean, std)

            # Compute different error metrics
            pos_error = np.sqrt(((pred_obs[:3] - true_obs[:3]) ** 2).sum())
            vel_error = np.sqrt(((pred_obs[3:6] - true_obs[3:6]) ** 2).sum())
            goal_error = np.sqrt(((pred_obs[6:] - true_obs[6:]) ** 2).sum())
            total_error = np.sqrt(((pred_obs - true_obs) ** 2).sum())
            norm_error = np.sqrt(((pred_obs_normalized - true_obs_normalized) ** 2).sum())

            episode_results['position_errors'].append(pos_error)
            episode_results['velocity_errors'].append(vel_error)
            episode_results['goal_errors'].append(goal_error)
            episode_results['total_errors'].append(total_error)
            episode_results['normalized_errors'].append(norm_error)

            if done:
                break

        # Add episode results
        for key in results.keys():
            results[key].append(episode_results[key])

    return results


def compute_statistics(results: Dict, horizon: int) -> Dict:
    """Compute mean, std, and percentiles for each timestep."""

    stats = {}

    for metric_name, episodes_data in results.items():
        # Find minimum common horizon
        min_horizon = min(len(ep) for ep in episodes_data if len(ep) > 0)
        min_horizon = min(min_horizon, horizon)

        # Compute statistics for each timestep
        means = []
        stds = []
        percentiles_25 = []
        percentiles_75 = []
        percentiles_95 = []

        for t in range(min_horizon):
            errors_at_t = [ep[t] for ep in episodes_data if len(ep) > t]

            means.append(np.mean(errors_at_t))
            stds.append(np.std(errors_at_t))
            percentiles_25.append(np.percentile(errors_at_t, 25))
            percentiles_75.append(np.percentile(errors_at_t, 75))
            percentiles_95.append(np.percentile(errors_at_t, 95))

        stats[metric_name] = {
            'mean': np.array(means),
            'std': np.array(stds),
            'p25': np.array(percentiles_25),
            'p75': np.array(percentiles_75),
            'p95': np.array(percentiles_95)
        }

    return stats


def find_effective_horizons(stats: Dict, thresholds: Dict[str, float]) -> Dict:
    """Find effective horizon for different error thresholds."""

    horizons = {}

    for metric_name, metric_stats in stats.items():
        threshold = thresholds.get(metric_name, 0.5)
        mean_errors = metric_stats['mean']

        # Find where error exceeds threshold
        effective_horizon = len(mean_errors)
        for t, error in enumerate(mean_errors):
            if error > threshold:
                effective_horizon = t
                break

        horizons[metric_name] = effective_horizon

    return horizons


def plot_comprehensive_results(stats: Dict, horizons: Dict, thresholds: Dict,
                              save_path: str = 'logs/rollout_comprehensive.png'):
    """Create comprehensive visualization of rollout results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('VAE + Latent Dynamics: Comprehensive Rollout Evaluation', fontsize=16)

    metrics = ['position_errors', 'velocity_errors', 'goal_errors',
               'total_errors', 'normalized_errors']
    titles = ['Position Error', 'Velocity Error', 'Goal Position Error',
              'Total State Error', 'Normalized Space Error']

    # Color scheme
    colors = sns.color_palette("husl", len(metrics))

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        metric_stats = stats[metric]
        horizon = horizons[metric]
        threshold = thresholds[metric]

        timesteps = np.arange(len(metric_stats['mean']))

        # Plot mean with confidence bands
        ax.plot(timesteps, metric_stats['mean'], color=colors[idx],
                linewidth=2, label='Mean')
        ax.fill_between(timesteps, metric_stats['p25'], metric_stats['p75'],
                        color=colors[idx], alpha=0.3, label='25-75 percentile')
        ax.fill_between(timesteps, metric_stats['p25'], metric_stats['p95'],
                        color=colors[idx], alpha=0.1, label='95 percentile')

        # Mark threshold and effective horizon
        ax.axhline(y=threshold, color='red', linestyle='--',
                  label=f'Threshold ({threshold:.2f})')
        ax.axvline(x=horizon, color='green', linestyle='--',
                  label=f'Effective horizon ({horizon})')

        # Target horizon
        ax.axvline(x=20, color='orange', linestyle=':', alpha=0.7,
                  label='Target (20 steps)')

        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('L2 Error')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim([0, min(50, len(metric_stats['mean']))])

    # Add summary statistics in the last subplot
    ax = axes[1, 2]
    summary_text = "Effective Horizons:\n" + "-"*25 + "\n"

    for metric, title in zip(metrics, titles):
        horizon = horizons[metric]
        threshold = thresholds[metric]
        status = "✓" if horizon >= 20 else "✗"
        summary_text += f"{title}:\n"
        summary_text += f"  Horizon: {horizon} steps {status}\n"
        summary_text += f"  Threshold: {threshold:.3f}\n\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComprehensive plot saved to {save_path}")

    return fig


def generate_trajectory_comparison(vae, dynamics, env, mean, std,
                                  n_trajectories: int = 3,
                                  horizon: int = 30):
    """Generate visual comparison of true vs predicted trajectories."""

    fig, axes = plt.subplots(n_trajectories, 3, figsize=(15, n_trajectories * 4))
    fig.suptitle('Trajectory Comparison: True vs Predicted', fontsize=16)

    for traj_idx in range(n_trajectories):
        # Reset environment
        true_obs = env.reset()

        # Generate action sequence
        actions = []
        for _ in range(horizon):
            action = np.random.uniform(-1, 1, size=env.action_dim)
            actions.append(action)
        actions = torch.FloatTensor(np.array(actions))

        # Collect true trajectory
        true_positions = [true_obs[:3].copy()]
        true_velocities = [true_obs[3:6].copy()]

        # Normalize and encode initial state
        obs_normalized = normalize(true_obs, mean, std)
        with torch.no_grad():
            z0 = vae.encode(torch.FloatTensor(obs_normalized).unsqueeze(0))

        # Collect predicted trajectory
        pred_positions = [true_obs[:3].copy()]  # Start from same position
        pred_velocities = [true_obs[3:6].copy()]

        z = z0
        for t in range(horizon):
            # True step
            true_obs, _, done, _ = env.step(actions[t].numpy())
            true_positions.append(true_obs[:3].copy())
            true_velocities.append(true_obs[3:6].copy())

            # Predicted step
            with torch.no_grad():
                z_next, _ = dynamics(z, actions[t].unsqueeze(0))
                z = z_next
                pred_obs_normalized = vae.decode(z).squeeze(0).numpy()

            pred_obs = denormalize(pred_obs_normalized, mean, std)
            pred_positions.append(pred_obs[:3].copy())
            pred_velocities.append(pred_obs[3:6].copy())

            if done:
                break

        true_positions = np.array(true_positions)
        pred_positions = np.array(pred_positions)
        true_velocities = np.array(true_velocities)
        pred_velocities = np.array(pred_velocities)

        # Plot 3D trajectory
        ax = axes[traj_idx, 0]
        ax.plot(true_positions[:, 0], true_positions[:, 1],
               'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(pred_positions[:, 0], pred_positions[:, 1],
               'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax.scatter(true_positions[0, 0], true_positions[0, 1],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(true_positions[-1, 0], true_positions[-1, 1],
                  c='blue', s=100, marker='s', label='End (true)')
        ax.scatter(pred_positions[-1, 0], pred_positions[-1, 1],
                  c='red', s=100, marker='^', label='End (pred)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Trajectory {traj_idx+1}: XY Plane')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot position errors over time
        ax = axes[traj_idx, 1]
        pos_errors = np.sqrt(((true_positions - pred_positions) ** 2).sum(axis=1))
        ax.plot(pos_errors, 'b-', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Position Error')
        ax.set_title(f'Position Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot velocity magnitude
        ax = axes[traj_idx, 2]
        true_vel_mag = np.sqrt((true_velocities ** 2).sum(axis=1))
        pred_vel_mag = np.sqrt((pred_velocities ** 2).sum(axis=1))
        ax.plot(true_vel_mag, 'b-', linewidth=2, label='True velocity', alpha=0.7)
        ax.plot(pred_vel_mag, 'r--', linewidth=2, label='Predicted velocity', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Velocity Magnitude')
        ax.set_title(f'Velocity Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'logs/trajectory_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTrajectory comparison saved to {save_path}")

    return fig


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE ROLLOUT EVALUATION")
    print("="*70)

    # Paths
    vae_path = 'weights/best_vae.pt'
    dynamics_path = 'weights/latent_world_model_vae.pt'

    # Check if models exist
    if not Path(vae_path).exists() or not Path(dynamics_path).exists():
        print("\n⚠️  Models not found. Training required first.")
        print(f"  VAE: {vae_path} - {'exists' if Path(vae_path).exists() else 'missing'}")
        print(f"  Dynamics: {dynamics_path} - {'exists' if Path(dynamics_path).exists() else 'missing'}")
        return

    # Load models
    print("\nLoading models...")
    vae, dynamics = load_models(vae_path, dynamics_path)
    print(f"  ✓ VAE loaded from {vae_path}")
    print(f"  ✓ Dynamics loaded from {dynamics_path}")

    # Load normalization stats
    mean, std = load_normalization_stats()
    print(f"  ✓ Normalization stats loaded")

    # Create environment
    env = Simple3DNavEnv()

    # Define error thresholds for different metrics
    thresholds = {
        'position_errors': 1.0,      # Position threshold
        'velocity_errors': 0.5,      # Velocity threshold
        'goal_errors': 2.0,          # Goal position threshold
        'total_errors': 2.0,         # Total state threshold
        'normalized_errors': 0.5     # Normalized space threshold
    }

    # Evaluate rollout quality
    results = evaluate_rollout_quality(
        vae, dynamics, env, mean, std,
        horizon=50, n_episodes=100
    )

    # Compute statistics
    stats = compute_statistics(results, horizon=50)

    # Find effective horizons
    horizons = find_effective_horizons(stats, thresholds)

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print("\nEffective Planning Horizons:")
    print("-" * 40)

    metrics_names = {
        'position_errors': 'Position',
        'velocity_errors': 'Velocity',
        'goal_errors': 'Goal Position',
        'total_errors': 'Total State',
        'normalized_errors': 'Normalized Space'
    }

    for metric, name in metrics_names.items():
        horizon = horizons[metric]
        threshold = thresholds[metric]
        status = "✓ PASSED" if horizon >= 20 else "✗ FAILED"

        print(f"\n{name}:")
        print(f"  Effective horizon: {horizon} steps")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Status: {status}")

        # Show errors at key timesteps
        if metric in stats:
            mean_errors = stats[metric]['mean']
            if len(mean_errors) > 0:
                print(f"  Error at step 5: {mean_errors[4] if len(mean_errors) > 4 else 'N/A':.4f}")
                print(f"  Error at step 10: {mean_errors[9] if len(mean_errors) > 9 else 'N/A':.4f}")
                print(f"  Error at step 20: {mean_errors[19] if len(mean_errors) > 19 else 'N/A':.4f}")

    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Comprehensive error plots
    plot_comprehensive_results(stats, horizons, thresholds)

    # Trajectory comparisons
    generate_trajectory_comparison(vae, dynamics, env, mean, std)

    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    normalized_horizon = horizons['normalized_errors']
    total_horizon = horizons['total_errors']

    if normalized_horizon >= 20 and total_horizon >= 20:
        print("\n🎉 SUCCESS! The model achieves the target planning horizon of 20+ steps")
        print(f"   Normalized space horizon: {normalized_horizon} steps")
        print(f"   Total state horizon: {total_horizon} steps")
    else:
        print("\n⚠️  Target horizon not fully achieved")
        print(f"   Normalized space: {normalized_horizon}/20 steps")
        print(f"   Total state: {total_horizon}/20 steps")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()