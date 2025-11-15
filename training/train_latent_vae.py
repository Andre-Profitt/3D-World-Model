#!/usr/bin/env python3
"""
Train Latent World Model using improved VAE encoder.

This version uses the trained VAE to encode observations into latent space,
then trains a dynamics model to predict transitions in that space.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VAE
from models.simple_latent_dynamics import SimpleLatentDynamics


class LatentTransitionDataset(Dataset):
    """Dataset for latent space transitions."""

    def __init__(self, data_path: str, vae_model: nn.Module, device: str = 'cpu'):
        """
        Load data and encode to latent space using VAE.

        Args:
            data_path: Path to transitions data (npz format)
            vae_model: Trained VAE model for encoding
            device: Device for computation
        """
        # Load raw data
        raw_data = np.load(data_path)

        # Move VAE to device and set to eval mode
        vae_model = vae_model.to(device)
        vae_model.eval()

        # Extract arrays from npz
        observations = raw_data['observations']  # [N, obs_dim]
        actions = raw_data['actions']  # [N, action_dim]
        rewards = raw_data['rewards']  # [N,]
        dones = raw_data.get('dones', np.zeros(len(observations)))  # [N,]

        # For transitions, we need current and next observations
        # Assuming sequential data, next obs is obs[1:]
        obs_current = observations[:-1]
        obs_next = observations[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        dones = dones[:-1]

        n_transitions = len(obs_current)

        # Encode observations to latent space
        self.latent_states = []
        self.next_latent_states = []

        print(f"Encoding {n_transitions} transitions to latent space...")
        batch_size = 256  # Process in batches for efficiency

        with torch.no_grad():
            for i in tqdm(range(0, n_transitions, batch_size), desc="Encoding"):
                batch_end = min(i + batch_size, n_transitions)

                # Get batch
                obs_batch = torch.FloatTensor(obs_current[i:batch_end]).to(device)
                next_obs_batch = torch.FloatTensor(obs_next[i:batch_end]).to(device)

                # Encode to latent (use mean, not sampled)
                z_batch = vae_model.encode(obs_batch, sample=False)
                z_next_batch = vae_model.encode(next_obs_batch, sample=False)

                self.latent_states.append(z_batch.cpu().numpy())
                self.next_latent_states.append(z_next_batch.cpu().numpy())

        # Concatenate batches
        self.latent_states = np.concatenate(self.latent_states, axis=0).astype(np.float32)
        self.next_latent_states = np.concatenate(self.next_latent_states, axis=0).astype(np.float32)
        self.actions = np.array(actions, dtype=np.float32)
        self.rewards = np.array(rewards, dtype=np.float32)
        self.dones = np.array(dones, dtype=np.float32)

        print(f"Encoded {len(self)} transitions to {self.latent_states.shape[1]}D latent space")

    def __len__(self):
        return len(self.latent_states)

    def __getitem__(self, idx):
        return {
            'state': self.latent_states[idx],
            'action': self.actions[idx],
            'next_state': self.next_latent_states[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx],
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_state_loss = 0
    total_reward_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move to device
        state = batch['state'].to(device)
        action = batch['action'].to(device)
        next_state = batch['next_state'].to(device)
        reward = batch['reward'].to(device)

        # Forward pass
        pred_next_state, pred_reward = model(state, action)

        # Compute losses
        state_loss = nn.MSELoss()(pred_next_state, next_state)
        reward_loss = nn.MSELoss()(pred_reward, reward.unsqueeze(-1))

        # Total loss with weighting
        loss = state_loss + 0.1 * reward_loss  # Weight reward less

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_state_loss += state_loss.item()
        total_reward_loss += reward_loss.item()

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'state_loss': total_state_loss / n_batches,
        'reward_loss': total_reward_loss / n_batches,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_state_loss = 0
    total_reward_loss = 0
    total_state_error = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move to device
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            next_state = batch['next_state'].to(device)
            reward = batch['reward'].to(device)

            # Forward pass
            pred_next_state, pred_reward = model(state, action)

            # Compute losses
            state_loss = nn.MSELoss()(pred_next_state, next_state)
            reward_loss = nn.MSELoss()(pred_reward, reward.unsqueeze(-1))

            # Total loss
            loss = state_loss + 0.1 * reward_loss

            # L2 error for interpretability
            state_error = torch.sqrt(((pred_next_state - next_state) ** 2).sum(dim=-1)).mean()

            # Track losses
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            total_state_error += state_error.item()

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'state_loss': total_state_loss / n_batches,
        'reward_loss': total_reward_loss / n_batches,
        'state_l2_error': total_state_error / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Latent World Model with VAE')
    parser.add_argument('--data_path', type=str, default='data/raw_trajectories.npz')
    parser.add_argument('--vae_path', type=str, default='models/checkpoints/vae_best.pth')
    parser.add_argument('--save_path', type=str, default='weights/latent_world_model_vae.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print(f"Training Latent World Model with VAE encoding")
    print(f"Device: {args.device}")

    # Load VAE model
    print(f"\nLoading VAE from {args.vae_path}...")
    checkpoint = torch.load(args.vae_path, map_location=args.device)

    # Extract VAE config from checkpoint
    vae_config = checkpoint.get('config', {})
    obs_dim = vae_config.get('obs_dim', 9)
    latent_dim = vae_config.get('latent_dim', 24)

    # Create VAE model
    vae = VAE(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        encoder_hidden=vae_config.get('encoder_hidden', [256, 256, 256]),
        decoder_hidden=vae_config.get('decoder_hidden', [256, 256, 256]),
        activation=vae_config.get('activation', 'elu'),
        beta=vae_config.get('beta', 0.001),
    )

    # Load weights
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    print(f"VAE loaded: {obs_dim}D → {latent_dim}D latent")

    # Create dataset with VAE encoding
    print(f"\nLoading and encoding data from {args.data_path}...")
    dataset = LatentTransitionDataset(args.data_path, vae, args.device)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create latent dynamics model
    print(f"\nCreating Latent Dynamics Model...")
    print(f"Architecture: {latent_dim}D + 3D action → {args.hidden_dim} × {args.n_layers} → {latent_dim}D + reward")

    model = SimpleLatentDynamics(
        latent_dim=latent_dim,
        action_dim=3,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(args.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device)

        # Evaluate
        val_metrics = evaluate(model, val_loader, args.device)

        # Step scheduler
        scheduler.step(val_metrics['loss'])

        # Print metrics
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"State: {train_metrics['state_loss']:.4f}, "
              f"Reward: {train_metrics['reward_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"State: {val_metrics['state_loss']:.4f}, "
              f"Reward: {val_metrics['reward_loss']:.4f}, "
              f"L2: {val_metrics['state_l2_error']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'latent_dim': latent_dim,
                'action_dim': 3,
                'hidden_dim': args.hidden_dim,
                'n_layers': args.n_layers,
                'vae_path': args.vae_path,
            }, args.save_path)

            print(f"  ✓ New best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {args.patience} epochs)")
                break

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()