#!/usr/bin/env python3
"""
Train VAE with scheduled beta (KL weight) for improved training stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.vae_scheduled import create_scheduled_vae


class ObservationDataset(Dataset):
    """Dataset for raw observations."""

    def __init__(self, observations):
        self.observations = torch.FloatTensor(observations)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx]


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    # Set total steps for scheduling
    model.set_training_steps(total_epochs * len(dataloader))

    progress_bar = tqdm(dataloader, desc=f'Training')
    for batch in progress_bar:
        batch = batch.to(device)

        # Forward pass with current step
        current_step = epoch * len(dataloader) + progress_bar.n
        reconstruction, mu, log_var, z = model(batch, step=current_step)

        # Compute loss
        total_loss_batch, recon_loss, kl_loss = model.loss(
            batch, reconstruction, mu, log_var, step=current_step
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track losses
        total_loss += total_loss_batch.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        # Update progress bar
        current_beta = model.get_current_beta()
        progress_bar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'beta': f'{current_beta:.5f}'
        })

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches,
        'beta': model.get_current_beta()
    }


def evaluate(model, dataloader, device, epoch, total_epochs):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Use current step for consistent beta
            current_step = epoch * len(dataloader)
            reconstruction, mu, log_var, z = model(batch, step=current_step)

            # Compute loss
            total_loss_batch, recon_loss, kl_loss = model.loss(
                batch, reconstruction, mu, log_var, step=current_step
            )

            total_loss += total_loss_batch.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train VAE with beta scheduling')
    parser.add_argument('--data_path', type=str, default='data/raw_trajectories.npz',
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                       help='Learning rate (from LR finder)')
    parser.add_argument('--schedule_type', type=str, default='cosine',
                       choices=['constant', 'linear', 'cosine', 'cyclical', 'monotonic'],
                       help='Beta scheduling type')
    parser.add_argument('--initial_beta', type=float, default=0.0001,
                       help='Initial beta value')
    parser.add_argument('--final_beta', type=float, default=0.01,
                       help='Final beta value')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps for scheduling')
    parser.add_argument('--free_bits', type=float, default=2.0,
                       help='Minimum KL per dimension')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    data = np.load(args.data_path)
    observations = data['observations']

    # Normalize data
    mean = observations.mean(axis=0)
    std = observations.std(axis=0) + 1e-8
    observations_norm = (observations - mean) / std

    # Save normalization stats
    Path('weights').mkdir(exist_ok=True)
    with open('weights/normalization_stats_scheduled.json', 'w') as f:
        json.dump({
            'mean': mean.tolist(),
            'std': std.tolist()
        }, f)

    # Create datasets
    n_train = int(0.9 * len(observations_norm))
    train_dataset = ObservationDataset(observations_norm[:n_train])
    val_dataset = ObservationDataset(observations_norm[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    obs_dim = observations.shape[-1]
    model = create_scheduled_vae(
        obs_dim=obs_dim,
        latent_dim=args.latent_dim,
        encoder_hidden=[256, 256, 256],
        decoder_hidden=[256, 256, 256],
        schedule_type=args.schedule_type,
        initial_beta=args.initial_beta,
        final_beta=args.final_beta,
        warmup_steps=args.warmup_steps,
        free_bits=args.free_bits
    ).to(device)

    print(f"\nModel Architecture:")
    print(f"  VAE: {obs_dim}D → {args.latent_dim}D latent")
    print(f"  Beta Schedule: {args.schedule_type}")
    print(f"  Beta Range: [{args.initial_beta}, {args.final_beta}]")
    print(f"  Free Bits: {args.free_bits}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch-1, args.epochs)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, epoch-1, args.epochs)

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon']:.4f}, "
              f"KL: {train_metrics['kl']:.4f}, "
              f"Beta: {train_metrics['beta']:.5f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Recon: {val_metrics['recon']:.4f}, "
              f"KL: {val_metrics['kl']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_metrics['recon'] < best_val_loss:
            best_val_loss = val_metrics['recon']
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'obs_dim': obs_dim,
                    'latent_dim': args.latent_dim,
                    'encoder_hidden': [256, 256, 256],
                    'decoder_hidden': [256, 256, 256],
                    'schedule_type': args.schedule_type,
                    'initial_beta': args.initial_beta,
                    'final_beta': args.final_beta,
                    'activation': 'elu'
                },
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, 'weights/best_vae_scheduled.pt')

            print(f"  ✓ New best model saved (val recon: {val_metrics['recon']:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️ Early stopping triggered (patience: {args.patience})")
            break

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            current_beta = model.get_current_beta()
            print(f"\n📊 Progress Report:")
            print(f"  Current Beta: {current_beta:.5f}")
            print(f"  Best Val Recon: {best_val_loss:.4f}")
            print(f"  Patience Counter: {patience_counter}/{args.patience}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final Val Recon: {val_metrics['recon']:.4f}")
    print(f"Best Val Recon: {best_val_loss:.4f}")
    print(f"Final Beta: {model.get_current_beta():.5f}")
    print("="*60)


if __name__ == "__main__":
    main()