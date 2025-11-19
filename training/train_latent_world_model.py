"""
Training script for latent world model.

Trains dynamics model in the compressed latent space learned by the autoencoder.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Optional

from models import Encoder, Decoder
from models.latent_world_model import LatentWorldModel, LatentEnsembleWorldModel, StochasticLatentWorldModel
# Import local config directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wm_config as project_config
config = project_config


class LatentTransitionDataset(Dataset):
    """Dataset for latent world model training."""

    def __init__(self, data_path: str, encoder: nn.Module, device: str = "cpu"):
        """
        Initialize dataset with pre-encoded latent states.

        Args:
            data_path: Path to .npz file containing transitions
            encoder: Trained encoder for converting obs to latent
            device: Device for encoding
        """
        data = np.load(data_path)

        # Load raw data
        observations = torch.from_numpy(data["observations"]).float()
        actions = torch.from_numpy(data["actions"]).float()
        next_observations = torch.from_numpy(data["next_observations"]).float()
        rewards = torch.from_numpy(data["rewards"]).float()
        dones = torch.from_numpy(data["dones"]).float()

        # Encode observations to latent space (in batches for efficiency)
        print("Encoding observations to latent space...")
        encoder = encoder.to(device)
        encoder.eval()

        batch_size = 1024
        latents = []
        next_latents = []

        with torch.no_grad():
            for i in tqdm(range(0, len(observations), batch_size)):
                batch_obs = observations[i:i+batch_size].to(device)
                batch_next_obs = next_observations[i:i+batch_size].to(device)

                batch_latent = encoder(batch_obs).cpu()
                batch_next_latent = encoder(batch_next_obs).cpu()

                latents.append(batch_latent)
                next_latents.append(batch_next_latent)

        self.latents = torch.cat(latents, dim=0)
        self.next_latents = torch.cat(next_latents, dim=0)
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

        # Store original observations for reconstruction loss (unused now but kept for reference)
        self.observations = observations
        self.next_observations = next_observations

        print(f"Loaded {len(self.latents)} transitions")
        print(f"Latent dimension: {self.latents.shape[-1]}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {
            "latent": self.latents[idx],
            "action": self.actions[idx],
            "next_latent": self.next_latents[idx],
            "reward": self.rewards[idx],
            "done": self.dones[idx],
            "obs": self.observations[idx],
            "next_obs": self.next_observations[idx],
        }


class LatentWorldModelTrainer:
    """Trainer for latent world model."""

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
        beta_dynamics: float = 1.0,
        beta_reward: float = 1.0,
    ):
        """
        Initialize trainer.

        Args:
            model: Latent world model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay
            gradient_clip: Gradient clipping value
            use_tensorboard: Whether to log to tensorboard
            beta_dynamics: Weight for dynamics loss
            beta_reward: Weight for reward loss
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip = gradient_clip

        # Loss weights
        self.beta_dynamics = beta_dynamics
        self.beta_reward = beta_reward

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated
            eta_min=1e-5,
        )

        # Logging
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            log_dir = config.LOGS_DIR / "tensorboard" / "latent_world_model"
            log_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir)

        # Best model tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_dynamics_loss = 0
        total_reward_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            latent = batch["latent"].to(self.device)
            action = batch["action"].to(self.device)
            next_latent = batch["next_latent"].to(self.device)
            reward = batch["reward"].to(self.device)

            # Forward pass and compute loss
            losses = self.model.loss(
                latent, action, next_latent, reward,
                beta_dynamics=self.beta_dynamics,
                beta_reward=self.beta_reward,
            )

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
            if "dynamics_loss" in losses:
                total_dynamics_loss += losses["dynamics_loss"].item()
            elif "nll_loss" in losses:
                total_dynamics_loss += losses["nll_loss"].item()
                
            total_reward_loss += losses["reward_loss"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": total_loss / num_batches,
                "dyn": total_dynamics_loss / num_batches,
                "rew": total_reward_loss / num_batches,
            })

            # Log to tensorboard
            if self.use_tensorboard and batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/loss", losses["loss"], global_step)
                if "dynamics_loss" in losses:
                    self.writer.add_scalar("train/dynamics_loss", losses["dynamics_loss"], global_step)
                elif "nll_loss" in losses:
                    self.writer.add_scalar("train/nll_loss", losses["nll_loss"], global_step)
                self.writer.add_scalar("train/reward_loss", losses["reward_loss"], global_step)

        return {
            "loss": total_loss / num_batches,
            "dynamics_loss": total_dynamics_loss / num_batches,
            "reward_loss": total_reward_loss / num_batches,
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0
        total_dynamics_loss = 0
        total_reward_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                # Move to device
                latent = batch["latent"].to(self.device)
                action = batch["action"].to(self.device)
                next_latent = batch["next_latent"].to(self.device)
                reward = batch["reward"].to(self.device)

                # Forward pass
                losses = self.model.loss(
                    latent, action, next_latent, reward,
                    beta_dynamics=self.beta_dynamics,
                    beta_reward=self.beta_reward,
                )

                # Track losses
                total_loss += losses["loss"].item()
                if "dynamics_loss" in losses:
                    total_dynamics_loss += losses["dynamics_loss"].item()
                elif "nll_loss" in losses:
                    total_dynamics_loss += losses["nll_loss"].item()
                total_reward_loss += losses["reward_loss"].item()
                num_batches += 1

        metrics = {
            "loss": total_loss / num_batches,
            "dynamics_loss": total_dynamics_loss / num_batches,
            "reward_loss": total_reward_loss / num_batches,
        }

        # Log to tensorboard
        if self.use_tensorboard:
            self.writer.add_scalar("val/loss", metrics["loss"], epoch)
            self.writer.add_scalar("val/dynamics_loss", metrics["dynamics_loss"], epoch)
            self.writer.add_scalar("val/reward_loss", metrics["reward_loss"], epoch)

        return metrics

    def train(self, num_epochs: int):
        """Train the model."""
        print("\n" + "="*50)
        print("Starting Latent World Model Training")
        print("="*50)
        print(f"Latent dim: {self.model.latent_dim}")
        print(f"Action dim: {self.model.action_dim}")
        print(f"Predict delta: {self.model.predict_delta}")

        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % 2 == 0:  # Validate every 2 epochs
                val_metrics = self.validate(epoch)

                print(f"\nEpoch {epoch}/{num_epochs}:")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"Dyn: {train_metrics['dynamics_loss']:.4f}, "
                      f"Rew: {train_metrics['reward_loss']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Dyn: {val_metrics['dynamics_loss']:.4f}, "
                      f"Rew: {val_metrics['reward_loss']:.4f}")

                # Check for improvement
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  New best model! Val loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= 15:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Update learning rate
            self.scheduler.step()

            # Log learning rate
            if self.use_tensorboard:
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch)

        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*50)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "latent_dim": self.model.latent_dim,
            "action_dim": self.model.action_dim,
        }

        # Determine filename based on model type
        if isinstance(self.model, StochasticLatentWorldModel):
            filename = "latent_world_model_stochastic.pt"
            best_filename = "best_latent_world_model_stochastic.pt"
        else:
            filename = "latent_world_model.pt"
            best_filename = "best_latent_world_model.pt"

        if is_best:
            path = config.WEIGHTS_DIR / best_filename
        else:
            path = config.WEIGHTS_DIR / filename

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")


def main():
    """Main training routine."""
    parser = argparse.ArgumentParser(description="Train latent world model")
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=str(config.MODEL_PATHS["encoder"]),
        help="Path to trained encoder"
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        default=str(config.MODEL_PATHS["decoder"]),
        help="Path to trained decoder"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--beta_recon",
        type=float,
        default=0.1,
        help="Weight for reconstruction loss (deprecated)"
    )
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        help="Train ensemble of latent models"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Ensemble size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE_CONFIG["device"],
        help="Device to train on"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Train stochastic world model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Set random seed to {args.seed}")

    # Check if encoder/decoder exist
    if not Path(args.encoder_path).exists():
        print(f"Error: Encoder not found at {args.encoder_path}")
        print("Please train the autoencoder first: python training/train_autoencoder.py")
        return

    # Load encoder (needed for dataset creation)
    print("Loading encoder...")
    encoder_checkpoint = torch.load(args.encoder_path, map_location=args.device)
    
    obs_dim = encoder_checkpoint["obs_dim"]
    latent_dim = encoder_checkpoint["latent_dim"]
    action_dim = 3  # From environment

    # Create encoder
    from models import Encoder
    encoder = Encoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=[128, 128],
        activation="relu",
        layer_norm=True,
    )
    encoder.load_state_dict(encoder_checkpoint["model_state_dict"])

    print(f"Loaded encoder (obs_dim={obs_dim}, latent_dim={latent_dim})")

    # Create datasets with pre-encoded latents
    print("\nCreating datasets...")
    train_dataset = LatentTransitionDataset(
        config.DATA_PATHS["train_data"],
        encoder,
        device=args.device
    )
    val_dataset = LatentTransitionDataset(
        config.DATA_PATHS["val_data"],
        encoder,
        device=args.device
    )

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

    # Create model
    if args.use_ensemble:
        print(f"\nCreating ensemble latent world model with {args.ensemble_size} members...")
        model = LatentEnsembleWorldModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            ensemble_size=args.ensemble_size,
            hidden_dims=[128, 128],
            activation="relu",
            predict_delta=True,
            separate_reward_head=True,
        )
    elif args.stochastic:
        print("\nCreating stochastic latent world model...")
        model = StochasticLatentWorldModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],
            activation="relu",
            predict_delta=True,
            separate_reward_head=True,
        )
    else:
        print("\nCreating latent world model...")
        model = LatentWorldModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=[128, 128],
            activation="relu",
            predict_delta=True,
            separate_reward_head=True,
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    dynamics_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable (dynamics) parameters: {dynamics_params:,}")

    # Create trainer
    trainer = LatentWorldModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=1e-5,
        gradient_clip=1.0,
        use_tensorboard=config.EXPERIMENT_CONFIG["use_tensorboard"],
        beta_dynamics=1.0,
        beta_reward=1.0,
    )

    # Train
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()