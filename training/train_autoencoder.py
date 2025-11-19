"""
Training script for observation autoencoder.

Learns compressed latent representations of observations for efficient dynamics modeling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Tuple

from models import Autoencoder
# Import local config directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wm_config as project_config
config = project_config


class ObservationDataset(Dataset):
    """Dataset for autoencoder training using collected observations."""

    def __init__(self, data_path: str):
        """
        Initialize dataset.

        Args:
            data_path: Path to .npz file containing observations
        """
        data = np.load(data_path)

        # Use observations and next_observations for more data
        obs = data["observations"]
        next_obs = data["next_observations"]

        # Combine for more training samples
        all_obs = np.concatenate([obs, next_obs], axis=0)

        # Remove duplicates (approximately)
        # Simple approach: shuffle and take unique-ish samples
        np.random.shuffle(all_obs)

        self.observations = torch.from_numpy(all_obs).float()
        print(f"Loaded {len(self.observations)} observations from {data_path}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx]


class AutoencoderTrainer:
    """Trainer for observation autoencoder."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        use_tensorboard: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: Autoencoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay
            gradient_clip: Gradient clipping value
            use_tensorboard: Whether to log to tensorboard
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip = gradient_clip

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.TRAINING_CONFIG["world_model"]["num_epochs"],
            eta_min=config.TRAINING_CONFIG["world_model"]["lr_scheduler"]["min_lr"],
        )

        # Logging
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            log_dir = config.LOGS_DIR / "tensorboard" / "autoencoder"
            log_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir)

        # Best model tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_latent_reg = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, obs in enumerate(pbar):
            # Move to device
            obs = obs.to(self.device)

            # Forward pass and compute loss
            losses = self.model.loss(obs)

            # Backward pass
            self.optimizer.zero_grad()
            losses["loss"].backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Track losses
            total_loss += losses["loss"].item()
            total_recon_loss += losses["recon_loss"].item()
            total_latent_reg += losses["latent_reg"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": total_loss / num_batches,
                "recon": total_recon_loss / num_batches,
            })

            # Log to tensorboard
            if self.use_tensorboard and batch_idx % config.TRAINING_CONFIG["log_frequency"] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/loss", losses["loss"], global_step)
                self.writer.add_scalar("train/recon_loss", losses["recon_loss"], global_step)
                self.writer.add_scalar("train/latent_reg", losses["latent_reg"], global_step)

        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "latent_reg": total_latent_reg / num_batches,
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_latent_reg = 0
        num_batches = 0

        # Track latent statistics
        all_latents = []

        with torch.no_grad():
            for obs in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                # Move to device
                obs = obs.to(self.device)

                # Forward pass
                reconstruction, latent = self.model(obs)
                losses = self.model.loss(obs)

                # Track losses
                total_loss += losses["loss"].item()
                total_recon_loss += losses["recon_loss"].item()
                total_latent_reg += losses["latent_reg"].item()
                num_batches += 1

                # Collect latents for analysis
                all_latents.append(latent.cpu())

        # Compute latent statistics
        all_latents = torch.cat(all_latents, dim=0)
        latent_mean = all_latents.mean(dim=0)
        latent_std = all_latents.std(dim=0)

        metrics = {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "latent_reg": total_latent_reg / num_batches,
            "latent_mean_norm": latent_mean.norm().item(),
            "latent_std_mean": latent_std.mean().item(),
        }

        # Log to tensorboard
        if self.use_tensorboard:
            self.writer.add_scalar("val/loss", metrics["loss"], epoch)
            self.writer.add_scalar("val/recon_loss", metrics["recon_loss"], epoch)
            self.writer.add_scalar("val/latent_reg", metrics["latent_reg"], epoch)
            self.writer.add_scalar("val/latent_mean_norm", metrics["latent_mean_norm"], epoch)
            self.writer.add_scalar("val/latent_std_mean", metrics["latent_std_mean"], epoch)

            # Log latent distribution
            self.writer.add_histogram("val/latent_mean", latent_mean, epoch)
            self.writer.add_histogram("val/latent_std", latent_std, epoch)

        return metrics

    def train(self, num_epochs: int):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*50)
        print("Starting Autoencoder Training")
        print("="*50)
        print(f"Observation dim: {self.model.obs_dim}")
        print(f"Latent dim: {self.model.latent_dim}")

        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % config.TRAINING_CONFIG["eval_frequency"] == 0:
                val_metrics = self.validate(epoch)

                print(f"\nEpoch {epoch}/{num_epochs}:")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Recon: {train_metrics['recon_loss']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Recon: {val_metrics['recon_loss']:.4f}")
                print(f"  Latent - Mean norm: {val_metrics['latent_mean_norm']:.3f}, "
                      f"Std: {val_metrics['latent_std_mean']:.3f}")

                # Check for improvement
                if val_metrics["recon_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["recon_loss"]
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  New best model! Val recon loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= config.TRAINING_CONFIG["world_model"]["early_stopping"]["patience"]:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

            # Periodic checkpoint
            if epoch % config.TRAINING_CONFIG["save_frequency"] == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Update learning rate
            self.scheduler.step()

            # Log learning rate
            if self.use_tensorboard:
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch)

        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best validation reconstruction loss: {self.best_val_loss:.4f}")
        print("="*50)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        # Save full autoencoder
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "obs_dim": self.model.obs_dim,
            "latent_dim": self.model.latent_dim,
        }

        if is_best:
            path = config.WEIGHTS_DIR / "best_autoencoder.pt"
        else:
            path = config.WEIGHTS_DIR / "autoencoder.pt"

        torch.save(checkpoint, path)
        print(f"  Saved autoencoder checkpoint to {path}")

        # Save encoder and decoder separately for modular use
        encoder_path = config.MODEL_PATHS["encoder"]
        decoder_path = config.MODEL_PATHS["decoder"]

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.encoder.state_dict(),
            "obs_dim": self.model.obs_dim,
            "latent_dim": self.model.latent_dim,
        }, encoder_path)

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.decoder.state_dict(),
            "obs_dim": self.model.obs_dim,
            "latent_dim": self.model.latent_dim,
        }, decoder_path)

        if is_best:
            print(f"  Saved encoder to {encoder_path}")
            print(f"  Saved decoder to {decoder_path}")


def visualize_reconstructions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_samples: int = 5,
):
    """
    Visualize reconstruction quality.

    Args:
        model: Trained autoencoder
        data_loader: Data loader
        device: Device
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt

    model.eval()
    samples = []
    reconstructions = []
    latents = []

    with torch.no_grad():
        for i, obs in enumerate(data_loader):
            if i >= num_samples:
                break

            obs = obs.to(device)
            recon, latent = model(obs)

            samples.append(obs[0].cpu().numpy())
            reconstructions.append(recon[0].cpu().numpy())
            latents.append(latent[0].cpu().numpy())

    # Create comparison plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # Original
        axes[i, 0].bar(range(len(samples[i])), samples[i])
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].set_ylabel("Value")

        # Reconstruction
        axes[i, 1].bar(range(len(reconstructions[i])), reconstructions[i])
        axes[i, 1].set_title(f"Reconstruction {i+1}")

        # Error
        error = samples[i] - reconstructions[i]
        axes[i, 2].bar(range(len(error)), error)
        axes[i, 2].set_title(f"Error (MSE: {np.mean(error**2):.4f})")

    plt.tight_layout()
    save_path = config.LOGS_DIR / "autoencoder_reconstructions.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved reconstruction visualization to {save_path}")
    plt.close()

    # Visualize latent space
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Latent activations
    latent_matrix = np.stack(latents)
    im = axes[0].imshow(latent_matrix, aspect='auto', cmap='coolwarm')
    axes[0].set_xlabel("Latent Dimension")
    axes[0].set_ylabel("Sample")
    axes[0].set_title("Latent Activations")
    plt.colorbar(im, ax=axes[0])

    # Latent statistics
    latent_mean = latent_matrix.mean(axis=0)
    latent_std = latent_matrix.std(axis=0)
    x = range(len(latent_mean))
    axes[1].bar(x, latent_mean, yerr=latent_std, capsize=3)
    axes[1].set_xlabel("Latent Dimension")
    axes[1].set_ylabel("Activation")
    axes[1].set_title("Latent Statistics (Mean ± Std)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.LOGS_DIR / "latent_space_analysis.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved latent space visualization to {save_path}")
    plt.close()


def main():
    """Main training routine."""
    parser = argparse.ArgumentParser(description="Train observation autoencoder")
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=config.MODEL_CONFIG["encoder"]["latent_dim"],
        help="Latent dimension"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="standard",
        choices=["shallow", "standard", "deep", "wide"],
        help="Architecture preset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.TRAINING_CONFIG["world_model"]["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,  # Fewer epochs for autoencoder
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.TRAINING_CONFIG["world_model"]["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize reconstructions after training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE_CONFIG["device"],
        help="Device to train on"
    )

    args = parser.parse_args()

    # Check if data exists
    if not config.DATA_PATHS["train_data"].exists():
        print(f"Error: Training data not found at {config.DATA_PATHS['train_data']}")
        print("Please run 'python scripts/collect_data.py' first")
        return

    # Create datasets
    train_dataset = ObservationDataset(config.DATA_PATHS["train_data"])
    val_dataset = ObservationDataset(config.DATA_PATHS["val_data"])

    # Get observation dimension
    obs_dim = train_dataset[0].shape[0]
    print(f"Observation dimension: {obs_dim}")
    print(f"Target latent dimension: {args.latent_dim}")
    print(f"Compression ratio: {obs_dim / args.latent_dim:.1f}x")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.DEVICE_CONFIG["num_workers"],
        pin_memory=config.DEVICE_CONFIG["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.DEVICE_CONFIG["num_workers"],
        pin_memory=config.DEVICE_CONFIG["pin_memory"],
    )

    # Architecture presets
    architectures = {
        "shallow": {"encoder_hidden": [64], "decoder_hidden": [64]},
        "standard": {"encoder_hidden": [128, 128], "decoder_hidden": [128, 128]},
        "deep": {"encoder_hidden": [256, 128, 64], "decoder_hidden": [64, 128, 256]},
        "wide": {"encoder_hidden": [512, 256], "decoder_hidden": [256, 512]},
    }

    arch = architectures[args.architecture]

    # Create autoencoder
    model = Autoencoder(
        obs_dim=obs_dim,
        latent_dim=args.latent_dim,
        encoder_hidden_dims=arch["encoder_hidden"],
        decoder_hidden_dims=arch["decoder_hidden"],
        activation="relu",
        layer_norm=True,
        dropout=0.0,
    )

    print(f"Created {args.architecture} autoencoder")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"Total parameters: {num_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")

    # Create trainer
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=config.TRAINING_CONFIG["world_model"]["weight_decay"],
        gradient_clip=config.TRAINING_CONFIG["world_model"]["gradient_clip"],
        use_tensorboard=config.EXPERIMENT_CONFIG["use_tensorboard"],
    )

    # Train
    trainer.train(num_epochs=args.num_epochs)

    # Visualize reconstructions
    if args.visualize:
        print("\nVisualizing reconstructions...")
        visualize_reconstructions(model, val_loader, args.device, num_samples=5)


if __name__ == "__main__":
    main()