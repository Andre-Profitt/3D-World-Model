#!/usr/bin/env python3
"""
Model comparison evaluation script.
Compares different model architectures and configurations.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
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


class ModelEvaluator:
    """Evaluate a single model configuration."""

    def __init__(self, name: str, vae_path: str = None, dynamics_path: str = None):
        self.name = name
        self.vae = None
        self.dynamics = None
        self.mean = None
        self.std = None

        if vae_path and Path(vae_path).exists():
            self.load_models(vae_path, dynamics_path)

    def load_models(self, vae_path: str, dynamics_path: str = None):
        """Load VAE and optionally dynamics model."""
        # Load VAE
        vae_checkpoint = torch.load(vae_path, map_location='cpu', weights_only=True)
        vae_config = vae_checkpoint.get('config', {})

        self.vae = VAE(
            obs_dim=vae_config.get('obs_dim', 9),
            latent_dim=vae_config.get('latent_dim', 24),
            encoder_hidden=vae_config.get('encoder_hidden', [256, 256, 256]),
            decoder_hidden=vae_config.get('decoder_hidden', [256, 256, 256]),
            activation=vae_config.get('activation', 'elu'),
            beta=vae_config.get('beta', 0.001),
        )
        self.vae.load_state_dict(vae_checkpoint['model_state_dict'])
        self.vae.eval()

        # Load dynamics if provided
        if dynamics_path and Path(dynamics_path).exists():
            dynamics_checkpoint = torch.load(dynamics_path, map_location='cpu', weights_only=True)

            self.dynamics = SimpleLatentDynamics(
                latent_dim=dynamics_checkpoint['latent_dim'],
                action_dim=dynamics_checkpoint['action_dim'],
                hidden_dim=dynamics_checkpoint['hidden_dim'],
                n_layers=dynamics_checkpoint['n_layers'],
            )
            self.dynamics.load_state_dict(dynamics_checkpoint['model_state_dict'])
            self.dynamics.eval()

        # Load normalization stats
        self.mean, self.std = load_normalization_stats()

    def evaluate_reconstruction(self, env, n_samples: int = 1000) -> Dict:
        """Evaluate reconstruction quality."""
        if self.vae is None:
            return None

        errors = []
        normalized_errors = []

        for _ in range(n_samples):
            # Random state
            obs = env.reset()
            for _ in range(np.random.randint(0, 20)):
                action = np.random.uniform(-1, 1, size=env.action_dim)
                obs, _, done, _ = env.step(action)
                if done:
                    obs = env.reset()

            # Normalize and reconstruct
            obs_normalized = normalize(obs, self.mean, self.std)
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)

            with torch.no_grad():
                recon_normalized, _, _, _ = self.vae(obs_tensor)

            # Compute errors
            recon = denormalize(recon_normalized.squeeze(0).numpy(), self.mean, self.std)
            error = np.sqrt(((recon - obs) ** 2).sum())
            normalized_error = np.sqrt(((recon_normalized.squeeze(0).numpy() - obs_normalized) ** 2).sum())

            errors.append(error)
            normalized_errors.append(normalized_error)

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_normalized_error': np.mean(normalized_errors),
            'std_normalized_error': np.std(normalized_errors),
            'p95_error': np.percentile(errors, 95),
            'p95_normalized_error': np.percentile(normalized_errors, 95)
        }

    def evaluate_rollout(self, env, horizon: int = 50, n_episodes: int = 100) -> Dict:
        """Evaluate rollout prediction quality."""
        if self.vae is None or self.dynamics is None:
            return None

        all_errors = []

        for _ in range(n_episodes):
            # Reset environment
            true_obs = env.reset()

            # Generate action sequence
            actions = torch.FloatTensor(np.random.uniform(-1, 1, (horizon, env.action_dim)))

            # Encode initial state
            obs_normalized = normalize(true_obs, self.mean, self.std)
            with torch.no_grad():
                z = self.vae.encode(torch.FloatTensor(obs_normalized).unsqueeze(0))

            episode_errors = []

            for t in range(horizon):
                # Predict next state
                with torch.no_grad():
                    z_next, _ = self.dynamics(z, actions[t].unsqueeze(0))
                    z = z_next
                    pred_obs_normalized = self.vae.decode(z).squeeze(0).numpy()

                # True next state
                true_obs, _, done, _ = env.step(actions[t].numpy())
                true_obs_normalized = normalize(true_obs, self.mean, self.std)

                # Compute error
                error = np.sqrt(((pred_obs_normalized - true_obs_normalized) ** 2).sum())
                episode_errors.append(error)

                if done:
                    break

            all_errors.append(episode_errors)

        # Compute statistics at different horizons
        horizons_to_check = [5, 10, 15, 20, 25, 30]
        stats = {}

        for h in horizons_to_check:
            errors_at_h = []
            for episode_errors in all_errors:
                if len(episode_errors) > h - 1:
                    errors_at_h.append(episode_errors[h - 1])

            if errors_at_h:
                stats[f'error_at_{h}'] = {
                    'mean': np.mean(errors_at_h),
                    'std': np.std(errors_at_h),
                    'p95': np.percentile(errors_at_h, 95)
                }

        # Find effective horizon
        threshold = 0.5
        effective_horizon = 0
        for episode_errors in all_errors:
            for t, error in enumerate(episode_errors):
                if error > threshold:
                    break
                effective_horizon = max(effective_horizon, t + 1)

        stats['effective_horizon'] = effective_horizon

        return stats


def compare_models(models: List[ModelEvaluator], env) -> Dict:
    """Compare multiple models."""

    results = {}

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    for model in models:
        print(f"\nEvaluating: {model.name}")
        print("-" * 40)

        model_results = {}

        # Reconstruction evaluation
        if model.vae is not None:
            print("  Testing reconstruction quality...")
            recon_results = model.evaluate_reconstruction(env, n_samples=500)
            model_results['reconstruction'] = recon_results

            print(f"    Mean error: {recon_results['mean_normalized_error']:.6f}")
            print(f"    95th percentile: {recon_results['p95_normalized_error']:.6f}")

        # Rollout evaluation
        if model.dynamics is not None:
            print("  Testing rollout quality...")
            rollout_results = model.evaluate_rollout(env, horizon=30, n_episodes=50)
            model_results['rollout'] = rollout_results

            if 'error_at_10' in rollout_results:
                print(f"    Error at step 10: {rollout_results['error_at_10']['mean']:.4f}")
            if 'error_at_20' in rollout_results:
                print(f"    Error at step 20: {rollout_results['error_at_20']['mean']:.4f}")
            print(f"    Effective horizon: {rollout_results['effective_horizon']} steps")

        results[model.name] = model_results

    return results


def plot_model_comparison(results: Dict, save_path: str = 'logs/model_comparison.png'):
    """Create visualization comparing different models."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Architecture Comparison', fontsize=16)

    models = list(results.keys())
    colors = sns.color_palette("husl", len(models))

    # 1. Reconstruction Error Comparison
    ax = axes[0, 0]
    recon_means = []
    recon_p95s = []
    valid_models = []

    for model_name in models:
        if 'reconstruction' in results[model_name]:
            recon = results[model_name]['reconstruction']
            recon_means.append(recon['mean_normalized_error'])
            recon_p95s.append(recon['p95_normalized_error'])
            valid_models.append(model_name)

    if recon_means:
        x = np.arange(len(valid_models))
        width = 0.35
        # Use safe color indexing
        color1 = colors[0] if len(colors) > 0 else 'blue'
        color2 = colors[1] if len(colors) > 1 else 'orange'
        ax.bar(x - width/2, recon_means, width, label='Mean', color=color1)
        ax.bar(x + width/2, recon_p95s, width, label='95th percentile', color=color2)
        ax.set_xlabel('Model')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Reconstruction Quality')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Effective Horizon Comparison
    ax = axes[0, 1]
    horizons = []
    horizon_models = []

    for model_name in models:
        if 'rollout' in results[model_name]:
            rollout = results[model_name]['rollout']
            if 'effective_horizon' in rollout:
                horizons.append(rollout['effective_horizon'])
                horizon_models.append(model_name)

    if horizons:
        ax.bar(horizon_models, horizons, color=colors[:len(horizon_models)])
        ax.axhline(y=20, color='red', linestyle='--', label='Target (20 steps)')
        ax.set_xlabel('Model')
        ax.set_ylabel('Effective Horizon (steps)')
        ax.set_title('Planning Horizon Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Error Growth Over Time
    ax = axes[1, 0]
    horizons = [5, 10, 15, 20, 25, 30]

    for i, model_name in enumerate(models):
        if 'rollout' in results[model_name]:
            rollout = results[model_name]['rollout']
            means = []
            for h in horizons:
                key = f'error_at_{h}'
                if key in rollout:
                    means.append(rollout[key]['mean'])
                else:
                    break

            if means:
                ax.plot(horizons[:len(means)], means, marker='o',
                       label=model_name, color=colors[i], linewidth=2)

    ax.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Mean Error')
    ax.set_title('Error Growth Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    # Create summary table data
    table_data = []
    headers = ['Model', 'Recon Error', 'Horizon', 'Error@10', 'Error@20']

    for model_name in models:
        row = [model_name[:15]]  # Truncate long names

        # Reconstruction error
        if 'reconstruction' in results[model_name]:
            recon = results[model_name]['reconstruction']
            row.append(f"{recon['mean_normalized_error']:.4f}")
        else:
            row.append('N/A')

        # Effective horizon
        if 'rollout' in results[model_name]:
            rollout = results[model_name]['rollout']
            row.append(f"{rollout.get('effective_horizon', 'N/A')}")

            # Error at specific steps
            if 'error_at_10' in rollout:
                row.append(f"{rollout['error_at_10']['mean']:.3f}")
            else:
                row.append('N/A')

            if 'error_at_20' in rollout:
                row.append(f"{rollout['error_at_20']['mean']:.3f}")
            else:
                row.append('N/A')
        else:
            row.extend(['N/A', 'N/A', 'N/A'])

        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color cells based on performance
    for i, row in enumerate(table_data):
        # Color horizon cell
        if row[2] != 'N/A':
            horizon_val = int(row[2])
            if horizon_val >= 20:
                table[(i+1, 2)].set_facecolor('#90EE90')  # Light green
            else:
                table[(i+1, 2)].set_facecolor('#FFB6C1')  # Light red

    ax.set_title('Summary Statistics')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")


def main():
    print("\n" + "="*60)
    print("MODEL COMPARISON EVALUATION")
    print("="*60)

    # Create environment
    env = Simple3DNavEnv()

    # Define models to compare
    models_to_compare = []

    # Current best model (VAE + Latent Dynamics)
    if Path('weights/best_vae.pt').exists():
        models_to_compare.append(
            ModelEvaluator(
                name="VAE + Latent Dynamics",
                vae_path='weights/best_vae.pt',
                dynamics_path='weights/latent_world_model_vae.pt'
            )
        )

    # Check for other model variants if they exist
    model_variants = [
        ('weights/vae_beta_0.01.pt', 'weights/dynamics_beta_0.01.pt', 'VAE (β=0.01)'),
        ('weights/vae_latent_16.pt', 'weights/dynamics_latent_16.pt', 'VAE (16D latent)'),
        ('weights/vae_latent_32.pt', 'weights/dynamics_latent_32.pt', 'VAE (32D latent)'),
    ]

    for vae_path, dyn_path, name in model_variants:
        if Path(vae_path).exists():
            models_to_compare.append(
                ModelEvaluator(
                    name=name,
                    vae_path=vae_path,
                    dynamics_path=dyn_path if Path(dyn_path).exists() else None
                )
            )

    if not models_to_compare:
        print("\n⚠️  No models found for comparison.")
        print("  Please train models first.")
        return

    # Run comparison
    results = compare_models(models_to_compare, env)

    # Create visualization
    plot_model_comparison(results)

    # Print best model
    print("\n" + "="*60)
    print("BEST MODEL SELECTION")
    print("="*60)

    best_horizon = 0
    best_model = None

    for model_name, model_results in results.items():
        if 'rollout' in model_results:
            horizon = model_results['rollout'].get('effective_horizon', 0)
            if horizon > best_horizon:
                best_horizon = horizon
                best_model = model_name

    if best_model:
        print(f"\n🏆 Best model: {best_model}")
        print(f"   Effective horizon: {best_horizon} steps")

        if 'reconstruction' in results[best_model]:
            recon = results[best_model]['reconstruction']
            print(f"   Reconstruction error: {recon['mean_normalized_error']:.6f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()