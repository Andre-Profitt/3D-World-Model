#!/usr/bin/env python3
"""
Evaluate the complete VAE + Latent Dynamics system.
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

sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VAE
from models.simple_latent_dynamics import SimpleLatentDynamics
from envs.simple_3d_nav import Simple3DNavEnv


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


def test_reconstruction_quality(vae, env, n_samples=1000):
    """Test VAE reconstruction quality."""
    print("\n" + "="*60)
    print("TESTING VAE RECONSTRUCTION QUALITY")
    print("="*60)

    errors = []

    for _ in tqdm(range(n_samples), desc="Testing reconstruction"):
        # Random state
        obs = env.reset()
        for _ in range(np.random.randint(0, 20)):
            # Random action with clipping to [-1, 1]
            action = np.random.uniform(-1, 1, size=env.action_dim)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        # Convert observation to tensor
        obs = torch.FloatTensor(obs).unsqueeze(0)

        # Reconstruct
        with torch.no_grad():
            recon, _, _, _ = vae(obs)

        # Compute error
        error = torch.sqrt(((recon - obs) ** 2).sum()).item()
        errors.append(error)

    errors = np.array(errors)

    print(f"\nReconstruction Results:")
    print(f"  Mean L2 error: {errors.mean():.6f}")
    print(f"  Median L2 error: {np.median(errors):.6f}")
    print(f"  95th percentile: {np.percentile(errors, 95):.6f}")
    print(f"  Max error: {errors.max():.6f}")

    target = 0.02
    success_rate = (errors < target).mean() * 100
    print(f"\n  Target: < {target}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  ✓ PASSED" if errors.mean() < target else f"  ✗ FAILED")

    return errors.mean()


def test_rollout_horizon(vae, dynamics, env, horizon=50, n_episodes=100):
    """Test effective planning horizon."""
    print("\n" + "="*60)
    print("TESTING EFFECTIVE PLANNING HORIZON")
    print("="*60)

    all_errors = []

    for episode in tqdm(range(n_episodes), desc="Testing rollouts"):
        # Reset environment
        true_obs = env.reset()

        # Generate random action sequence
        actions = []
        for _ in range(horizon):
            action = np.random.uniform(-1, 1, size=env.action_dim)
            actions.append(action)
        actions = torch.FloatTensor(np.array(actions))

        # Encode initial state
        with torch.no_grad():
            z0 = vae.encode(torch.FloatTensor(true_obs).unsqueeze(0))

        # Rollout in latent space
        episode_errors = []
        z = z0

        for t in range(horizon):
            # Predict next latent
            with torch.no_grad():
                z_next, _ = dynamics(z, actions[t].unsqueeze(0))
                z = z_next

            # Decode to observation
            with torch.no_grad():
                pred_obs = vae.decode(z).squeeze(0).numpy()

            # True next state
            true_obs, _, done, _ = env.step(actions[t].numpy())

            # Compute error
            error = np.sqrt(((pred_obs - true_obs) ** 2).sum())
            episode_errors.append(error)

            if done:
                break

        all_errors.append(episode_errors)

    # Analyze horizon
    max_horizon = min(len(e) for e in all_errors)
    mean_errors = []

    for t in range(max_horizon):
        errors_at_t = [e[t] for e in all_errors if len(e) > t]
        mean_errors.append(np.mean(errors_at_t))

    # Find effective horizon (where error exceeds threshold)
    threshold = 0.5  # Reasonable error threshold
    effective_horizon = max_horizon

    for t, error in enumerate(mean_errors):
        if error > threshold:
            effective_horizon = t
            break

    print(f"\nRollout Results:")
    print(f"  Tested horizon: {horizon} steps")
    print(f"  Error at step 5: {mean_errors[4] if len(mean_errors) > 4 else 'N/A':.4f}")
    print(f"  Error at step 10: {mean_errors[9] if len(mean_errors) > 9 else 'N/A':.4f}")
    print(f"  Error at step 20: {mean_errors[19] if len(mean_errors) > 19 else 'N/A':.4f}")
    print(f"  Error at step 30: {mean_errors[29] if len(mean_errors) > 29 else 'N/A':.4f}")

    print(f"\n  Effective horizon (error < {threshold}): {effective_horizon} steps")
    print(f"  Target: > 20 steps")
    print(f"  ✓ PASSED" if effective_horizon > 20 else f"  ✗ FAILED")

    # Plot error growth
    plt.figure(figsize=(10, 6))
    plt.plot(mean_errors[:min(50, len(mean_errors))], 'b-', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.axvline(x=20, color='g', linestyle='--', label='Target horizon (20)')
    if effective_horizon < len(mean_errors):
        plt.axvline(x=effective_horizon, color='orange', linestyle='--',
                   label=f'Effective horizon ({effective_horizon})')
    plt.xlabel('Prediction Step')
    plt.ylabel('Mean L2 Error')
    plt.title('VAE + Latent Dynamics Rollout Error Growth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('logs/vae_latent_horizon.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to logs/vae_latent_horizon.png")

    return effective_horizon, mean_errors


def main():
    print("\n" + "="*60)
    print("VAE + LATENT DYNAMICS SYSTEM EVALUATION")
    print("="*60)

    # Paths
    vae_path = 'weights/best_vae.pt'
    dynamics_path = 'weights/latent_world_model_vae.pt'

    # Load models
    print("\nLoading models...")
    vae, dynamics = load_models(vae_path, dynamics_path)
    print(f"  ✓ VAE loaded from {vae_path}")
    print(f"  ✓ Dynamics loaded from {dynamics_path}")

    # Create environment
    env = Simple3DNavEnv()

    # Test 1: Reconstruction quality
    recon_error = test_reconstruction_quality(vae, env, n_samples=1000)

    # Test 2: Rollout horizon
    horizon, errors = test_rollout_horizon(vae, dynamics, env, horizon=50, n_episodes=100)

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print("\nTarget Metrics:")
    print("  1. Reconstruction error < 0.02")
    print(f"     Result: {recon_error:.6f} {'✓ PASSED' if recon_error < 0.02 else '✗ FAILED'}")

    print("\n  2. Effective horizon > 20 steps")
    print(f"     Result: {horizon} steps {'✓ PASSED' if horizon > 20 else '✗ FAILED'}")

    if recon_error < 0.02 and horizon > 20:
        print("\n🎉 ALL TARGET METRICS ACHIEVED! 🎉")
        print("\nThe improved VAE + Latent Dynamics system successfully meets all requirements:")
        print("- Excellent reconstruction quality")
        print("- Long-horizon planning capability")
        print("\nKey improvements over baseline:")
        print("- VAE reconstruction: 1.79 → {:.6f} ({}x better!)".format(
            recon_error, int(1.79 / recon_error)))
        print("- Effective horizon: ~3 → {} steps ({}x longer!)".format(
            horizon, int(horizon / 3)))
    else:
        print("\nSome targets not met. Further optimization needed.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()