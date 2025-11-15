"""
Improved VAE training with normalization, scheduling, and validation monitoring.

Key improvements:
- Data normalization and augmentation
- Learning rate scheduling with warmup
- Gradient clipping
- Early stopping with patience
- Better logging and visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import argparse

from models.vae import VAE
import config


class NormalizedDataset(Dataset):
    """Dataset with normalization and augmentation."""

    def __init__(
        self,
        data_path: Path,
        normalize: bool = True,
        augment: bool = True,
        noise_std: float = 0.01,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to data file
            normalize: Whether to normalize data
            augment: Whether to apply augmentation
            noise_std: Standard deviation of noise for augmentation
        """
        data = np.load(data_path)
        self.observations = data["observations"].astype(np.float32)

        self.normalize = normalize
        self.augment = augment
        self.noise_std = noise_std

        # Compute normalization statistics
        if self.normalize:
            self.mean = self.observations.mean(axis=0)
            self.std = self.observations.std(axis=0) + 1e-6  # Avoid division by zero
        else:
            self.mean = np.zeros(self.observations.shape[1])
            self.std = np.ones(self.observations.shape[1])

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx].copy()

        # Apply normalization
        if self.normalize:
            obs = (obs - self.mean) / self.std

        # Apply augmentation (small random noise)
        if self.augment and np.random.rand() > 0.5:
            obs += np.random.normal(0, self.noise_std, obs.shape).astype(np.float32)

        return torch.from_numpy(obs)

    def denormalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Denormalize observations."""
        if self.normalize:
            return obs * torch.from_numpy(self.std).to(obs.device) + torch.from_numpy(
                self.mean
            ).to(obs.device)
        return obs


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / self.base_lrs[0])

        return [base_lr * lr_scale for base_lr in self.base_lrs]


class VAETrainer:
    """Improved VAE trainer with monitoring and early stopping."""

    def __init__(
        self,
        model: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        patience: int = 20,
        gradient_clip: float = 1.0,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: VAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Initial learning rate
            max_epochs: Maximum training epochs
            warmup_epochs: Number of warmup epochs
            patience: Early stopping patience
            gradient_clip: Gradient clipping value
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
        )

        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.save_dir = save_dir or config.WEIGHTS_DIR
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch in pbar:
                batch = batch.to(self.device)

                # Forward pass
                recon, mu, logvar, z = self.model(batch)

                # Compute loss
                loss, components = self.model.loss(batch, recon, mu, logvar)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

                self.optimizer.step()

                # Track losses
                total_loss += components["total"]
                total_recon += components["recon"]
                total_kl += components["kl"]
                num_batches += 1

                # Update progress bar
                pbar.set_postfix(
                    loss=f"{components['total']:.4f}",
                    recon=f"{components['recon']:.4f}",
                    kl=f"{components['kl']:.4f}",
                )

        return {
            "total": total_loss / num_batches,
            "recon": total_recon / num_batches,
            "kl": total_kl / num_batches,
        }

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_l2 = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Forward pass
                recon, mu, logvar, z = self.model(batch)

                # Compute loss
                loss, components = self.model.loss(batch, recon, mu, logvar)

                # Also compute L2 reconstruction error
                l2_error = torch.norm(batch - recon, dim=1).mean()

                total_loss += components["total"]
                total_recon += components["recon"]
                total_kl += components["kl"]
                total_l2 += l2_error.item()
                num_batches += 1

        return {
            "total": total_loss / num_batches,
            "recon": total_recon / num_batches,
            "kl": total_kl / num_batches,
            "l2": total_l2 / num_batches,
        }

    def train(self) -> Dict[str, list]:
        """Complete training loop."""
        print(f"Starting VAE training for {self.max_epochs} epochs...")
        print(f"Model: {self.model.obs_dim}D → {self.model.latent_dim}D latent")
        print(f"Beta (KL weight): {self.model.beta}")

        for epoch in range(self.max_epochs):
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print(f"  Train - Loss: {train_metrics['total']:.4f}, "
                  f"Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}")
            print(f"  Val   - Loss: {val_metrics['total']:.4f}, "
                  f"Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f}, "
                  f"L2: {val_metrics['l2']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Check for improvement
            if val_metrics["recon"] < self.best_val_loss:
                self.best_val_loss = val_metrics["recon"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model (val recon: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered (patience: {self.patience})")
                    break

            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        # Save final model
        self.save_checkpoint(epoch, is_best=False, final=True)

        return {
            "train": self.train_losses,
            "val": self.val_losses,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": {
                "obs_dim": self.model.obs_dim,
                "latent_dim": self.model.latent_dim,
                "beta": self.model.beta,
            },
        }

        if is_best:
            path = self.save_dir / "best_vae.pt"
        elif final:
            path = self.save_dir / "final_vae.pt"
        else:
            path = self.save_dir / f"vae_epoch_{epoch + 1}.pt"

        torch.save(checkpoint, path)

    def plot_losses(self, save_path: Optional[Path] = None):
        """Plot training and validation losses."""
        epochs = range(1, len(self.train_losses) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Total loss
        ax = axes[0, 0]
        ax.plot(epochs, [l["total"] for l in self.train_losses], "b-", label="Train")
        ax.plot(epochs, [l["total"] for l in self.val_losses], "r-", label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total VAE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reconstruction loss
        ax = axes[0, 1]
        ax.plot(epochs, [l["recon"] for l in self.train_losses], "b-", label="Train")
        ax.plot(epochs, [l["recon"] for l in self.val_losses], "r-", label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction Loss")
        ax.set_title("Reconstruction Loss (MSE)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # KL loss
        ax = axes[1, 0]
        ax.plot(epochs, [l["kl"] for l in self.train_losses], "b-", label="Train")
        ax.plot(epochs, [l["kl"] for l in self.val_losses], "r-", label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # L2 reconstruction error (validation only)
        ax = axes[1, 1]
        ax.plot(epochs, [l["l2"] for l in self.val_losses], "g-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L2 Error")
        ax.set_title("Validation L2 Reconstruction Error")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.02, color="r", linestyle="--", label="Target (<0.02)")
        ax.legend()

        plt.suptitle("VAE Training Progress", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved loss plot to {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train improved VAE")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="Latent dimension",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Beta for KL weighting (beta-VAE)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE_CONFIG["device"],
        help="Device to train on",
    )

    args = parser.parse_args()

    # Create datasets
    print("Loading data...")
    train_dataset = NormalizedDataset(
        config.DATA_DIR / "train_data.npz",
        normalize=True,
        augment=True,
        noise_std=0.01,
    )

    val_dataset = NormalizedDataset(
        config.DATA_DIR / "val_data.npz",
        normalize=True,
        augment=False,  # No augmentation for validation
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    obs_dim = train_dataset.observations.shape[1]
    model = VAE(
        obs_dim=obs_dim,
        latent_dim=args.latent_dim,
        encoder_hidden=[256, 256, 256],
        decoder_hidden=[256, 256, 256],
        activation="elu",
        beta=args.beta,
    )

    print(f"\nModel architecture:")
    print(f"  Encoder: {obs_dim}D → [256, 256, 256] → {args.latent_dim}D")
    print(f"  Decoder: {args.latent_dim}D → [256, 256, 256] → {obs_dim}D")
    print(f"  Beta (KL weight): {args.beta}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        gradient_clip=1.0,
    )

    # Train model
    losses = trainer.train()

    # Plot losses
    plot_path = config.LOGS_DIR / "vae_training.png"
    trainer.plot_losses(save_path=plot_path)

    # Save normalization statistics
    stats = {
        "mean": train_dataset.mean.tolist(),
        "std": train_dataset.std.tolist(),
    }
    stats_path = config.WEIGHTS_DIR / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization statistics to {stats_path}")

    print("\nTraining complete!")
    print(f"Best validation reconstruction loss: {trainer.best_val_loss:.6f}")
    print(f"Final validation L2 error: {trainer.val_losses[-1]['l2']:.6f}")

    if trainer.val_losses[-1]["l2"] < 0.02:
        print("✓ Achieved target L2 error < 0.02!")
    else:
        print(f"⚠ L2 error {trainer.val_losses[-1]['l2']:.4f} > 0.02 target")


if __name__ == "__main__":
    main()