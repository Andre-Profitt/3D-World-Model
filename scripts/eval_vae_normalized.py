#!/usr/bin/env python3
"""
Evaluate the complete VAE + Latent Dynamics system with proper normalization.
Tests for target metrics:
- Reconstruction error < 0.02
- Effective horizon > 20 steps
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

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
    vae_checkpoint = torch.load(vae_path, map_location='cpu')
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
    dynamics_checkpoint = torch.load(dynamics_path, map_location='cpu')

    dynamics = SimpleLatentDynamics(
        latent_dim=dynamics_checkpoint['latent_dim'],
        action_dim=dynamics_checkpoint['action_dim'],
        hidden_dim=dynamics_checkpoint['hidden_dim'],
        n_layers=dynamics_checkpoint['n_layers'],
    )
    dynamics.load_state_dict(dynamics_checkpoint['model_state_dict'])
    dynamics.eval()

    return vae, dynamics


def test_reconstruction_quality(vae, env, mean, std, n_samples=1000):
    """Test VAE reconstruction quality with normalization."""
    print("\n" + "="*60)
    print("TESTING VAE RECONSTRUCTION QUALITY")
    print("="*60)

    errors = []
    normalized_errors = []

    for _ in tqdm(range(n_samples), desc="Testing reconstruction"):
        # Random state
        obs = env.reset()
        for _ in range(np.random.randint(0, 20)):
            action = np.random.uniform(-1, 1, size=env.action_dim)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        # Normalize observation
        obs_normalized = normalize(obs, mean, std)
        obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)

        # Reconstruct
        with torch.no_grad():
            recon_normalized, _, _, _ = vae(obs_tensor)

        # Denormalize reconstruction
        recon = denormalize(recon_normalized.squeeze(0).numpy(), mean, std)

        # Compute errors
        error = np.sqrt(((recon - obs) ** 2).sum())
        normalized_error = np.sqrt(((recon_normalized.squeeze(0).numpy() - obs_normalized) ** 2).sum())

        errors.append(error)
        normalized_errors.append(normalized_error)

    errors = np.array(errors)
    normalized_errors = np.array(normalized_errors)

    print(f"\nReconstruction Results (Original Space):")
    print(f"  Mean L2 error: {errors.mean():.6f}")
    print(f"  Median L2 error: {np.median(errors):.6f}")
    print(f"  95th percentile: {np.percentile(errors, 95):.6f}")

    print(f"\nReconstruction Results (Normalized Space):")
    print(f"  Mean L2 error: {normalized_errors.mean():.6f}")
    print(f"  Median L2 error: {np.median(normalized_errors):.6f}")
    print(f"  95th percentile: {np.percentile(normalized_errors, 95):.6f}")

    target = 0.02
    success_rate = (normalized_errors < target).mean() * 100
    print(f"\n  Target (normalized): < {target}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  ✓ PASSED" if normalized_errors.mean() < target else f"  ✗ FAILED")

    return normalized_errors.mean(), errors.mean()


def test_rollout_horizon(vae, dynamics, env, mean, std, horizon=50, n_episodes=100):
    """Test effective planning horizon with normalization."""
    print("\n" + "="*60)
    print("TESTING EFFECTIVE PLANNING HORIZON")
    print("="*60)

    all_errors = []
    all_normalized_errors = []

    for episode in tqdm(range(n_episodes), desc="Testing rollouts"):
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

        # Rollout in latent space
        episode_errors = []
        episode_normalized_errors = []
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

            # Compute errors
            error = np.sqrt(((pred_obs - true_obs) ** 2).sum())
            normalized_error = np.sqrt(((pred_obs_normalized - true_obs_normalized) ** 2).sum())

            episode_errors.append(error)
            episode_normalized_errors.append(normalized_error)

            if done:
                break

        all_errors.append(episode_errors)
        all_normalized_errors.append(episode_normalized_errors)

    # Analyze horizon
    max_horizon = min(len(e) for e in all_normalized_errors if len(e) > 0)
    mean_errors = []
    mean_normalized_errors = []

    for t in range(max_horizon):
        errors_at_t = [e[t] for e in all_errors if len(e) > t]
        normalized_errors_at_t = [e[t] for e in all_normalized_errors if len(e) > t]
        mean_errors.append(np.mean(errors_at_t))
        mean_normalized_errors.append(np.mean(normalized_errors_at_t))

    # Find effective horizon (where normalized error exceeds threshold)
    threshold = 0.5  # Reasonable error threshold in normalized space
    effective_horizon = max_horizon

    for t, error in enumerate(mean_normalized_errors):
        if error > threshold:
            effective_horizon = t
            break

    print(f"\nRollout Results (Original Space):")
    print(f"  Error at step 5: {mean_errors[4] if len(mean_errors) > 4 else 'N/A':.4f}")
    print(f"  Error at step 10: {mean_errors[9] if len(mean_errors) > 9 else 'N/A':.4f}")
    print(f"  Error at step 20: {mean_errors[19] if len(mean_errors) > 19 else 'N/A':.4f}")

    print(f"\nRollout Results (Normalized Space):")
    print(f"  Error at step 5: {mean_normalized_errors[4] if len(mean_normalized_errors) > 4 else 'N/A':.4f}")
    print(f"  Error at step 10: {mean_normalized_errors[9] if len(mean_normalized_errors) > 9 else 'N/A':.4f}")
    print(f"  Error at step 20: {mean_normalized_errors[19] if len(mean_normalized_errors) > 19 else 'N/A':.4f}")

    print(f"\n  Effective horizon (normalized error < {threshold}): {effective_horizon} steps")
    print(f"  Target: > 20 steps")
    print(f"  ✓ PASSED" if effective_horizon > 20 else f"  ✗ FAILED")

    # Plot error growth
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(mean_normalized_errors[:min(50, len(mean_normalized_errors))], 'b-', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.axvline(x=20, color='g', linestyle='--', label='Target horizon (20)')
    if effective_horizon < len(mean_normalized_errors):
        plt.axvline(x=effective_horizon, color='orange', linestyle='--',
                   label=f'Effective horizon ({effective_horizon})')
    plt.xlabel('Prediction Step')
    plt.ylabel('Mean L2 Error (Normalized)')
    plt.title('VAE + Latent Dynamics: Normalized Space')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(mean_errors[:min(50, len(mean_errors))], 'b-', linewidth=2)
    plt.xlabel('Prediction Step')
    plt.ylabel('Mean L2 Error (Original)')
    plt.title('VAE + Latent Dynamics: Original Space')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logs/vae_latent_horizon_normalized.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to logs/vae_latent_horizon_normalized.png")

    return effective_horizon, mean_normalized_errors


def main():
    print("\n" + "="*60)
    print("VAE + LATENT DYNAMICS SYSTEM EVALUATION (WITH NORMALIZATION)")
    print("="*60)

    # Paths
    vae_path = 'weights/best_vae.pt'
    dynamics_path = 'weights/latent_world_model_vae.pt'

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

    # Test 1: Reconstruction quality
    norm_recon_error, orig_recon_error = test_reconstruction_quality(vae, env, mean, std, n_samples=1000)

    # Test 2: Rollout horizon
    horizon, errors = test_rollout_horizon(vae, dynamics, env, mean, std, horizon=50, n_episodes=100)

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print("\nTarget Metrics:")
    print("  1. Reconstruction error < 0.02 (normalized space)")
    print(f"     Result: {norm_recon_error:.6f} {'✓ PASSED' if norm_recon_error < 0.02 else '✗ FAILED'}")
    print(f"     Original space: {orig_recon_error:.6f}")

    print("\n  2. Effective horizon > 20 steps")
    print(f"     Result: {horizon} steps {'✓ PASSED' if horizon > 20 else '✗ FAILED'}")

    if norm_recon_error < 0.02 and horizon > 20:
        print("\n🎉 ALL TARGET METRICS ACHIEVED! 🎉")
        print("\nThe improved VAE + Latent Dynamics system successfully meets all requirements:")
        print("- Excellent reconstruction quality in normalized space")
        print("- Long-horizon planning capability")
        print("\nKey improvements:")
        print("- VAE reconstruction (normalized): {:.6f} (target: <0.02)".format(norm_recon_error))
        print("- Effective horizon: {} steps (target: >20)".format(horizon))
    else:
        print("\nSome targets not met. Analysis:")
        if norm_recon_error >= 0.02:
            print("- Reconstruction needs improvement")
        if horizon <= 20:
            print("- Planning horizon needs extension")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()