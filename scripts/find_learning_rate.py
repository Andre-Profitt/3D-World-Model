#!/usr/bin/env python3
"""
Learning Rate Finder for automatic hyperparameter optimization.
Implements the technique from "Cyclical Learning Rates for Training Neural Networks" (Leslie Smith, 2015).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple, List, Optional
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VAE
from models.vae_scheduled import ScheduledVAE, create_scheduled_vae
from models.simple_latent_dynamics import SimpleLatentDynamics


class LearningRateFinder:
    """
    Learning rate finder for PyTorch models.
    Gradually increases learning rate and tracks loss to find optimal range.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Store original state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()

        # Results storage
        self.learning_rates = []
        self.losses = []
        self.smoothed_losses = []

    def find(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.98,
        diverge_th: float = 5.0
    ) -> Tuple[List[float], List[float]]:
        """
        Find optimal learning rate range.

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Maximum learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss
            diverge_th: Divergence threshold

        Returns:
            learning_rates: List of tested learning rates
            losses: List of corresponding losses
        """
        # Reset model and optimizer
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Move model to device
        self.model.to(self.device)
        self.model.train()

        # Calculate learning rate schedule
        lr_schedule = np.exp(np.linspace(np.log(start_lr), np.log(end_lr), num_iter))

        # Initialize
        iterator = iter(train_loader)
        best_loss = float('inf')
        smoothed_loss = 0

        print(f"Finding learning rate from {start_lr:.2e} to {end_lr:.2e}")
        progress_bar = tqdm(range(num_iter), desc="LR Finder")

        for iteration in progress_bar:
            # Get batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            # Set learning rate
            lr = lr_schedule[iteration]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            loss = self._train_batch(batch)

            # Smooth loss
            if iteration == 0:
                smoothed_loss = loss
            else:
                smoothed_loss = smooth_f * smoothed_loss + (1 - smooth_f) * loss

            # Record
            self.learning_rates.append(lr)
            self.losses.append(loss)
            self.smoothed_losses.append(smoothed_loss)

            progress_bar.set_postfix({'lr': f'{lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})

            # Check for divergence
            if smoothed_loss > diverge_th * best_loss:
                print(f"\n⚠️  Stopping early due to diverging loss")
                break

            # Update best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        return self.learning_rates, self.smoothed_losses

    def _train_batch(self, batch) -> float:
        """Train on a single batch and return loss."""
        if isinstance(batch, (tuple, list)):
            data = batch[0].to(self.device)
            if len(batch) > 1:
                target = batch[1].to(self.device)
            else:
                target = data
        else:
            data = batch.to(self.device)
            target = data

        self.optimizer.zero_grad()

        # Handle different model types
        if hasattr(self.model, 'forward'):
            output = self.model(data)

            # Handle VAE-style outputs
            if isinstance(output, tuple) and len(output) == 4:
                reconstruction, mu, log_var, z = output
                if hasattr(self.model, 'loss'):
                    loss, _, _ = self.model.loss(data, reconstruction, mu, log_var)
                else:
                    recon_loss = nn.functional.mse_loss(reconstruction, target)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon_loss + 0.001 * kl_loss
            else:
                loss = self.criterion(output, target)
        else:
            loss = self.criterion(self.model(data), target)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def plot(self, skip_start: int = 10, skip_end: int = 5, log_scale: bool = True) -> Tuple[float, float]:
        """
        Plot learning rate vs loss and suggest optimal range.

        Args:
            skip_start: Number of iterations to skip at start
            skip_end: Number of iterations to skip at end
            log_scale: Use log scale for learning rate axis

        Returns:
            suggested_min_lr: Suggested minimum learning rate
            suggested_max_lr: Suggested maximum learning rate
        """
        if len(self.learning_rates) == 0:
            print("No data to plot. Run find() first.")
            return None, None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data
        lrs = self.learning_rates[skip_start:-skip_end] if skip_end > 0 else self.learning_rates[skip_start:]
        losses = self.smoothed_losses[skip_start:-skip_end] if skip_end > 0 else self.smoothed_losses[skip_start:]

        # Plot
        ax.plot(lrs, losses, 'b-', linewidth=2)

        # Find suggested range
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]

        # Find steepest slope (maximum negative gradient)
        gradients = np.gradient(losses)
        steepest_idx = np.argmin(gradients)
        steepest_lr = lrs[steepest_idx]

        # Suggest range
        suggested_min_lr = steepest_lr / 10  # One order of magnitude before steepest
        suggested_max_lr = min_loss_lr / 10  # One order of magnitude before minimum

        # Mark suggested range
        ax.axvline(x=suggested_min_lr, color='g', linestyle='--', label=f'Min LR: {suggested_min_lr:.2e}')
        ax.axvline(x=suggested_max_lr, color='r', linestyle='--', label=f'Max LR: {suggested_max_lr:.2e}')

        # Labels and formatting
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_xscale('log')

        plt.tight_layout()
        save_path = 'logs/learning_rate_finder.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nLearning rate plot saved to {save_path}")

        print(f"\n📊 Suggested Learning Rate Range:")
        print(f"   Min: {suggested_min_lr:.2e}")
        print(f"   Max: {suggested_max_lr:.2e}")

        return suggested_min_lr, suggested_max_lr


def find_lr_for_vae(
    data_path: str = 'data/raw_trajectories.npz',
    latent_dim: int = 32,
    schedule_type: str = 'cosine'
) -> Tuple[float, float]:
    """Find optimal learning rate for VAE training."""

    print("\n" + "="*60)
    print("FINDING OPTIMAL LEARNING RATE FOR VAE")
    print("="*60)

    # Load data
    data = np.load(data_path)
    observations = data['observations']

    # Normalize data
    if Path('weights/normalization_stats.json').exists():
        with open('weights/normalization_stats.json', 'r') as f:
            norm_stats = json.load(f)
        mean = np.array(norm_stats['mean'])
        std = np.array(norm_stats['std'])
    else:
        mean = observations.mean(axis=0)
        std = observations.std(axis=0) + 1e-8

    observations_norm = (observations - mean) / std

    # Create dataset
    dataset = TensorDataset(torch.FloatTensor(observations_norm))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create model
    obs_dim = observations.shape[-1]
    model = create_scheduled_vae(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        encoder_hidden=[256, 256, 256],  # Default encoder architecture
        decoder_hidden=[256, 256, 256],  # Default decoder architecture
        schedule_type=schedule_type,
        initial_beta=0.0001,
        final_beta=0.01
    )

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-7)

    # Create criterion (will be handled internally by VAE)
    criterion = nn.MSELoss()

    # Create finder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finder = LearningRateFinder(model, optimizer, criterion, device)

    # Find learning rate
    lrs, losses = finder.find(
        dataloader,
        start_lr=1e-7,
        end_lr=1.0,
        num_iter=200
    )

    # Plot and get suggestions
    min_lr, max_lr = finder.plot()

    return min_lr, max_lr


def find_lr_for_dynamics(
    vae_path: str = 'weights/best_vae.pt',
    data_path: str = 'data/raw_trajectories.npz',
    hidden_dim: int = 512,
    n_layers: int = 4
) -> Tuple[float, float]:
    """Find optimal learning rate for dynamics model training."""

    print("\n" + "="*60)
    print("FINDING OPTIMAL LEARNING RATE FOR DYNAMICS MODEL")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VAE for encoding
    print("Loading VAE...")
    vae_checkpoint = torch.load(vae_path, map_location=device)
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

    # Load and encode data
    print("Loading and encoding data...")
    data = np.load(data_path)
    observations = data['observations']
    actions = data['actions']
    next_observations = data['next_observations']

    # Normalize
    with open('weights/normalization_stats.json', 'r') as f:
        norm_stats = json.load(f)
    mean = np.array(norm_stats['mean'])
    std = np.array(norm_stats['std'])

    observations_norm = (observations - mean) / (std + 1e-8)
    next_observations_norm = (next_observations - mean) / (std + 1e-8)

    # Encode to latent space
    with torch.no_grad():
        latent_states = vae.encode(torch.FloatTensor(observations_norm)).numpy()
        next_latent_states = vae.encode(torch.FloatTensor(next_observations_norm)).numpy()

    # Create dataset
    dataset = TensorDataset(
        torch.FloatTensor(latent_states),
        torch.FloatTensor(actions),
        torch.FloatTensor(next_latent_states)
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Create model
    latent_dim = vae_config.get('latent_dim', 24)
    action_dim = actions.shape[-1]

    model = SimpleLatentDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-7)

    # Create criterion
    def dynamics_criterion(output, target):
        if isinstance(output, tuple):
            next_state_pred, _ = output
            return nn.functional.mse_loss(next_state_pred, target)
        return nn.functional.mse_loss(output, target)

    # Modified train batch for dynamics
    class DynamicsFinder(LearningRateFinder):
        def _train_batch(self, batch) -> float:
            states, actions, next_states = batch
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            next_state_pred, _ = self.model(states, actions)

            # Compute loss
            loss = nn.functional.mse_loss(next_state_pred, next_states)

            loss.backward()
            self.optimizer.step()

            return loss.item()

    # Create finder
    finder = DynamicsFinder(model, optimizer, None, device)

    # Find learning rate
    lrs, losses = finder.find(
        dataloader,
        start_lr=1e-6,
        end_lr=0.1,
        num_iter=200
    )

    # Plot and get suggestions
    min_lr, max_lr = finder.plot()

    return min_lr, max_lr


def main():
    """Find optimal learning rates for all models."""

    import argparse
    parser = argparse.ArgumentParser(description='Find optimal learning rates')
    parser.add_argument('--model', type=str, default='all',
                       choices=['vae', 'dynamics', 'all'],
                       help='Which model to find LR for')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to config file')

    args = parser.parse_args()

    results = {}

    if args.model in ['vae', 'all']:
        min_lr, max_lr = find_lr_for_vae()
        results['vae'] = {
            'min_lr': float(min_lr),
            'max_lr': float(max_lr),
            'suggested_lr': float(np.sqrt(min_lr * max_lr))  # Geometric mean
        }

    if args.model in ['dynamics', 'all']:
        if Path('weights/best_vae.pt').exists():
            min_lr, max_lr = find_lr_for_dynamics()
            results['dynamics'] = {
                'min_lr': float(min_lr),
                'max_lr': float(max_lr),
                'suggested_lr': float(np.sqrt(min_lr * max_lr))
            }
        else:
            print("⚠️  VAE model not found. Train VAE first.")

    # Print summary
    print("\n" + "="*60)
    print("LEARNING RATE RECOMMENDATIONS")
    print("="*60)

    for model_name, lr_info in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Range: [{lr_info['min_lr']:.2e}, {lr_info['max_lr']:.2e}]")
        print(f"  Suggested: {lr_info['suggested_lr']:.2e}")

    # Save results if requested
    if args.save_results and results:
        config_path = 'configs/learning_rates.json'
        Path('configs').mkdir(exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {config_path}")

    print("="*60)


if __name__ == "__main__":
    main()