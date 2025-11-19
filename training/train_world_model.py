"""
Training script for the 3D world model.

Trains the model to predict next state and reward from current state and action.
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
from typing import Dict, Tuple

from models import WorldModel, EnsembleWorldModel
import wm_config as config


class TransitionDataset(Dataset):
    """Dataset for world model training."""

    def __init__(self, data_path: str):
        """
        Initialize dataset.

        Args:
            data_path: Path to .npz file containing data
        """
        data = np.load(data_path)
        self.observations = torch.from_numpy(data["observations"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()
        self.next_observations = torch.from_numpy(data["next_observations"]).float()
        self.rewards = torch.from_numpy(data["rewards"]).float()
        self.dones = torch.from_numpy(data["dones"]).float()

        print(f"Loaded {len(self.observations)} transitions from {data_path}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # Return a single transition
        return {
            "obs": self.observations[idx],
            "action": self.actions[idx],
            "next_obs": self.next_observations[idx],
            "reward": self.rewards[idx],
            "done": self.dones[idx],
        }

class SequenceDataset(Dataset):
    """Dataset for multi-step world model training."""
    
    def __init__(self, data_path: str, seq_len: int = 10):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to .npz file
            seq_len: Length of sequences to return
        """
        data = np.load(data_path)
        self.observations = torch.from_numpy(data["observations"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()
        self.rewards = torch.from_numpy(data["rewards"]).float()
        self.dones = torch.from_numpy(data["dones"]).float()
        
        self.seq_len = seq_len
        
        # Pre-compute valid start indices
        # A start index is valid if the sequence doesn't cross episode boundaries (done=True)
        self.valid_indices = []
        total_len = len(self.observations)
        
        # This is a bit slow for large datasets, but safe
        # Assuming data is stored as [ep1_t1, ep1_t2, ..., ep1_done, ep2_t1, ...]
        # We check if any 'done' flag appears in the sequence (except possibly the last step)
        
        # Faster way: identify episode boundaries
        # done_indices = np.where(data["dones"])[0]
        # But let's stick to a simple loop for now or just random sampling during training if dataset is huge.
        # Given the data size, we can probably iterate.
        
        for i in range(total_len - seq_len):
            # Check if any step in the sequence (except the last) is terminal
            # If i+k is terminal, then i+k+1 belongs to next episode.
            # So we can't have a transition from i+k to i+k+1.
            # We need obs[i]...obs[i+seq_len] to be in same episode.
            # dones[j] being true means transition j -> j+1 is valid but j+1 is terminal state (or start of next?)
            # Usually done[t] corresponds to transition (s_t, a_t) -> s_{t+1}. If done[t] is true, s_{t+1} is terminal.
            # So we can use up to the step where done is True.
            
            # If any done in [i, i+seq_len-2] is True, then the sequence is broken.
            if not torch.any(self.dones[i:i+seq_len-1]):
                self.valid_indices.append(i)
                
        print(f"Found {len(self.valid_indices)} valid sequences of length {seq_len}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        
        return {
            "obs_seq": self.observations[start_idx:end_idx],
            "action_seq": self.actions[start_idx:end_idx], # Note: last action might not be useful if we don't have next obs
            "reward_seq": self.rewards[start_idx:end_idx],
        }


class WorldModelTrainer:
    """Trainer for world model."""

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
            model: World model to train
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
            log_dir = config.LOGS_DIR / "tensorboard"
            log_dir.mkdir(exist_ok=True)
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
        total_state_loss = 0
        total_reward_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            # Forward pass
            if "obs_seq" in batch:
                # Multi-step loss
                obs_seq = batch["obs_seq"].to(self.device)
                action_seq = batch["action_seq"].to(self.device)
                reward_seq = batch["reward_seq"].to(self.device)
                
                losses = self.model.unrolled_loss(obs_seq, action_seq, reward_seq, unroll_steps=5)
            else:
                # Single-step loss
                obs = batch["obs"].to(self.device)
                action = batch["action"].to(self.device)
                next_obs = batch["next_obs"].to(self.device)
                reward = batch["reward"].to(self.device)
                
                losses = self.model.loss(obs, action, next_obs, reward)

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
            total_state_loss += losses["state_loss"].item()
            total_reward_loss += losses["reward_loss"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": total_loss / num_batches,
                "state": total_state_loss / num_batches,
                "reward": total_reward_loss / num_batches,
            })

            # Log to tensorboard
            if self.use_tensorboard and batch_idx % config.TRAINING_CONFIG["log_frequency"] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/loss", losses["loss"], global_step)
                self.writer.add_scalar("train/state_loss", losses["state_loss"], global_step)
                self.writer.add_scalar("train/reward_loss", losses["reward_loss"], global_step)

        return {
            "loss": total_loss / num_batches,
            "state_loss": total_state_loss / num_batches,
            "reward_loss": total_reward_loss / num_batches,
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
        total_state_loss = 0
        total_reward_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                # Move to device
                # Forward pass
                if "obs_seq" in batch:
                    # Multi-step loss
                    obs_seq = batch["obs_seq"].to(self.device)
                    action_seq = batch["action_seq"].to(self.device)
                    reward_seq = batch["reward_seq"].to(self.device)
                    
                    losses = self.model.unrolled_loss(obs_seq, action_seq, reward_seq, unroll_steps=5)
                else:
                    # Single-step loss
                    obs = batch["obs"].to(self.device)
                    action = batch["action"].to(self.device)
                    next_obs = batch["next_obs"].to(self.device)
                    reward = batch["reward"].to(self.device)
                    
                    losses = self.model.loss(obs, action, next_obs, reward)

                # Track losses
                total_loss += losses["loss"].item()
                total_state_loss += losses["state_loss"].item()
                total_reward_loss += losses["reward_loss"].item()
                num_batches += 1

        metrics = {
            "loss": total_loss / num_batches,
            "state_loss": total_state_loss / num_batches,
            "reward_loss": total_reward_loss / num_batches,
        }

        # Log to tensorboard
        if self.use_tensorboard:
            self.writer.add_scalar("val/loss", metrics["loss"], epoch)
            self.writer.add_scalar("val/state_loss", metrics["state_loss"], epoch)
            self.writer.add_scalar("val/reward_loss", metrics["reward_loss"], epoch)

        return metrics

    def train(self, num_epochs: int):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*50)
        print("Starting World Model Training")
        print("="*50)

        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % config.TRAINING_CONFIG["eval_frequency"] == 0:
                val_metrics = self.validate(epoch)

                print(f"\nEpoch {epoch}/{num_epochs}:")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"State: {train_metrics['state_loss']:.4f}, "
                      f"Reward: {train_metrics['reward_loss']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"State: {val_metrics['state_loss']:.4f}, "
                      f"Reward: {val_metrics['reward_loss']:.4f}")

                # Check for improvement
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  New best model! Val loss: {self.best_val_loss:.4f}")
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
        }

        if is_best:
            path = config.MODEL_PATHS["best_model"]
        else:
            path = config.MODEL_PATHS["world_model"]

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")


def main():
    """Main training routine."""
    parser = argparse.ArgumentParser(description="Train 3D world model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.TRAINING_CONFIG["world_model"]["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=config.TRAINING_CONFIG["world_model"]["num_epochs"],
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.TRAINING_CONFIG["world_model"]["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        default=config.MODEL_CONFIG.get("use_ensemble", False),
        help="Train ensemble model"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=config.MODEL_CONFIG.get("ensemble_size", 5),
        help="Ensemble size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE_CONFIG["device"],
        help="Device to train on"
    )
    parser.add_argument(
        "--use_multistep",
        action="store_true",
        help="Use multi-step training loss"
    )

    args = parser.parse_args()

    # Check if data exists
    if not config.DATA_PATHS["train_data"].exists():
        print(f"Error: Training data not found at {config.DATA_PATHS['train_data']}")
        print("Please run 'python scripts/collect_data.py' first")
        return

    # Create datasets
    if args.use_multistep:
        print("Using SequenceDataset for multi-step training")
        train_dataset = SequenceDataset(config.DATA_PATHS["train_data"], seq_len=10)
        val_dataset = SequenceDataset(config.DATA_PATHS["val_data"], seq_len=10)
    else:
        train_dataset = TransitionDataset(config.DATA_PATHS["train_data"])
        val_dataset = TransitionDataset(config.DATA_PATHS["val_data"])

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

    # Get dimensions from dataset
    sample = train_dataset[0]
    if args.use_multistep:
        obs_dim = sample["obs_seq"].shape[1]
        action_dim = sample["action_seq"].shape[1]
    else:
        obs_dim = sample["obs"].shape[0]
        action_dim = sample["action"].shape[0]

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    # Create model
    if args.use_ensemble:
        # For ensemble, we'll train each member separately with bootstrap sampling
        print(f"Training ensemble with {args.ensemble_size} members")

        for member_idx in range(args.ensemble_size):
            print(f"\n{'='*60}")
            print(f"Training Ensemble Member {member_idx + 1}/{args.ensemble_size}")
            print('='*60)

            # Create bootstrap dataset for this member
            bootstrap_indices = np.random.choice(
                len(train_dataset),
                size=int(len(train_dataset) * config.MODEL_CONFIG.get("bootstrap_ratio", 1.0)),
                replace=True
            )
            bootstrap_dataset = torch.utils.data.Subset(train_dataset, bootstrap_indices)

            # Create bootstrap data loader
            bootstrap_loader = DataLoader(
                bootstrap_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=config.DEVICE_CONFIG["num_workers"],
                pin_memory=config.DEVICE_CONFIG["pin_memory"],
            )

            # Create single model for this member
            model = WorldModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                **config.MODEL_CONFIG["world_model"],
            )

            # Create trainer
            trainer = WorldModelTrainer(
                model=model,
                train_loader=bootstrap_loader,
                val_loader=val_loader,
                device=args.device,
                lr=args.lr,
                weight_decay=config.TRAINING_CONFIG["world_model"]["weight_decay"],
                gradient_clip=config.TRAINING_CONFIG["world_model"]["gradient_clip"],
                use_tensorboard=config.EXPERIMENT_CONFIG["use_tensorboard"],
            )

            # Train this member
            trainer.train(num_epochs=args.num_epochs)

            # Save member checkpoint
            member_path = config.WEIGHTS_DIR / f"world_model_member_{member_idx}.pt"
            torch.save({
                "member_idx": member_idx,
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "config": config.MODEL_CONFIG["world_model"],
            }, member_path)
            print(f"Saved ensemble member {member_idx} to {member_path}")

    else:
        model = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **config.MODEL_CONFIG["world_model"],
        )
        print("Created single world model")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {num_params:,}")

        # Create trainer
        trainer = WorldModelTrainer(
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


if __name__ == "__main__":
    main()