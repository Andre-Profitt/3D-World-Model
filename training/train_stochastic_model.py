"""
Training script for stochastic world models.

Trains probabilistic dynamics models that output distributions over next states.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Tuple

from models import StochasticWorldModel, StochasticEnsembleWorldModel
import config


def load_data(data_path: Path) -> Tuple[np.ndarray, ...]:
    """Load training data from disk."""
    data = np.load(data_path)

    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]

    return observations, actions, next_observations, rewards


def create_datasets(
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    train_ratio: float = 0.9,
) -> Tuple[TensorDataset, TensorDataset]:
    """Create train and validation datasets."""

    # Convert to tensors
    obs_tensor = torch.FloatTensor(observations)
    action_tensor = torch.FloatTensor(actions)
    next_obs_tensor = torch.FloatTensor(next_observations)
    reward_tensor = torch.FloatTensor(rewards)

    # Create dataset
    full_dataset = TensorDataset(
        obs_tensor, action_tensor, next_obs_tensor, reward_tensor
    )

    # Split into train/val
    num_samples = len(full_dataset)
    num_train = int(num_samples * train_ratio)

    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    return train_dataset, val_dataset


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate stochastic model on validation set."""
    model.eval()

    total_loss = 0.0
    total_state_nll = 0.0
    total_reward_nll = 0.0
    total_state_entropy = 0.0
    total_reward_entropy = 0.0
    num_batches = 0

    # Also track deterministic prediction errors
    total_state_error = 0.0
    total_reward_error = 0.0

    with torch.no_grad():
        for batch in val_loader:
            obs, action, next_obs, reward = [x.to(device) for x in batch]

            # Get loss components
            loss_dict = model.loss(obs, action, next_obs, reward)

            total_loss += loss_dict["loss"].item()
            total_state_nll += loss_dict["state_nll"].item()
            total_reward_nll += loss_dict["reward_nll"].item()
            total_state_entropy += loss_dict["state_entropy"].item()
            total_reward_entropy += loss_dict["reward_entropy"].item()

            # Get deterministic predictions (mean)
            next_obs_pred, reward_pred, _ = model(
                obs, action, deterministic=True
            )

            state_error = torch.mean((next_obs_pred - next_obs) ** 2).item()
            reward_error = torch.mean((reward_pred.squeeze() - reward) ** 2).item()

            total_state_error += state_error
            total_reward_error += reward_error

            num_batches += 1

    return {
        "val_loss": total_loss / num_batches,
        "val_state_nll": total_state_nll / num_batches,
        "val_reward_nll": total_reward_nll / num_batches,
        "val_state_entropy": total_state_entropy / num_batches,
        "val_reward_entropy": total_reward_entropy / num_batches,
        "val_state_mse": total_state_error / num_batches,
        "val_reward_mse": total_reward_error / num_batches,
    }


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float = 1.0,
    scheduler_config: dict = None,
    log_frequency: int = 10,
    model_idx: int = 0,
) -> Dict[str, list]:
    """Train a single stochastic model."""

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Setup scheduler if configured
    scheduler = None
    if scheduler_config and scheduler_config.get("type") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=scheduler_config.get("min_lr", 1e-5)
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_state_nll": [],
        "val_reward_nll": [],
        "val_state_mse": [],
        "val_reward_mse": [],
    }

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_state_nll = 0.0
        epoch_reward_nll = 0.0
        num_batches = 0

        # Training loop
        progress_bar = tqdm(
            train_loader,
            desc=f"Model {model_idx} - Epoch {epoch+1}/{num_epochs}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            obs, action, next_obs, reward = [x.to(device) for x in batch]

            # Forward pass
            loss_dict = model.loss(obs, action, next_obs, reward)
            loss = loss_dict["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_state_nll += loss_dict["state_nll"].item()
            epoch_reward_nll += loss_dict["reward_nll"].item()
            num_batches += 1

            # Update progress bar
            if batch_idx % log_frequency == 0:
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "state_nll": loss_dict["state_nll"].item(),
                    "reward_nll": loss_dict["reward_nll"].item(),
                })

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)

        # Store history
        history["train_loss"].append(epoch_loss / num_batches)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_state_nll"].append(val_metrics["val_state_nll"])
        history["val_reward_nll"].append(val_metrics["val_reward_nll"])
        history["val_state_mse"].append(val_metrics["val_state_mse"])
        history["val_reward_mse"].append(val_metrics["val_reward_mse"])

        # Check for best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_model_state = model.state_dict().copy()

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Print epoch summary
        print(f"Model {model_idx} - Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_loss/num_batches:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val State NLL: {val_metrics['val_state_nll']:.4f}")
        print(f"  Val Reward NLL: {val_metrics['val_reward_nll']:.4f}")
        print(f"  Val State MSE: {val_metrics['val_state_mse']:.4f}")
        print(f"  Val Reward MSE: {val_metrics['val_reward_mse']:.4f}")
        print(f"  Val State Entropy: {val_metrics['val_state_entropy']:.3f}")
        print(f"  Val Reward Entropy: {val_metrics['val_reward_entropy']:.3f}")
        print()

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return history


def train_ensemble(
    obs_dim: int,
    action_dim: int,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    device: torch.device,
    ensemble_size: int,
    bootstrap_ratio: float,
    **training_kwargs
) -> Tuple[StochasticEnsembleWorldModel, list]:
    """Train an ensemble of stochastic models."""

    # Create ensemble
    ensemble = StochasticEnsembleWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        ensemble_size=ensemble_size,
        **config.MODEL_CONFIG.get("stochastic_world_model", {})
    ).to(device)

    histories = []

    for member_idx in range(ensemble_size):
        print(f"\n{'='*60}")
        print(f"Training Ensemble Member {member_idx + 1}/{ensemble_size}")
        print(f"{'='*60}\n")

        # Create bootstrap sample
        if bootstrap_ratio < 1.0:
            num_samples = int(len(train_dataset) * bootstrap_ratio)
            bootstrap_indices = np.random.choice(
                len(train_dataset),
                size=num_samples,
                replace=True
            )
            member_train_dataset = Subset(train_dataset, bootstrap_indices)
        else:
            member_train_dataset = train_dataset

        # Create data loader
        train_loader = DataLoader(
            member_train_dataset,
            batch_size=training_kwargs["batch_size"],
            shuffle=True,
            num_workers=config.DEVICE_CONFIG["num_workers"],
            pin_memory=config.DEVICE_CONFIG["pin_memory"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_kwargs["batch_size"],
            shuffle=False,
            num_workers=config.DEVICE_CONFIG["num_workers"],
            pin_memory=config.DEVICE_CONFIG["pin_memory"],
        )

        # Train this member
        history = train_single_model(
            model=ensemble.models[member_idx],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_idx=member_idx,
            **training_kwargs
        )

        histories.append(history)

    return ensemble, histories


def analyze_uncertainty(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> Dict[str, float]:
    """Analyze uncertainty decomposition of the model."""
    model.eval()

    all_epistemic_state = []
    all_aleatoric_state = []
    all_total_state = []
    all_epistemic_reward = []
    all_aleatoric_reward = []
    all_total_reward = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break

            obs, action, next_obs, reward = [x.to(device) for x in batch]

            if isinstance(model, StochasticEnsembleWorldModel):
                # Get uncertainty decomposition
                predictions, uncertainty_info, _ = model(
                    obs, action,
                    deterministic=False,
                    reduce="mean_std"
                )

                all_epistemic_state.append(
                    uncertainty_info["epistemic_state_std"].mean().item()
                )
                all_aleatoric_state.append(
                    uncertainty_info["aleatoric_state_std"].mean().item()
                )
                all_total_state.append(
                    uncertainty_info["total_state_std"].mean().item()
                )
                all_epistemic_reward.append(
                    uncertainty_info["epistemic_reward_std"].mean().item()
                )
                all_aleatoric_reward.append(
                    uncertainty_info["aleatoric_reward_std"].mean().item()
                )
                all_total_reward.append(
                    uncertainty_info["total_reward_std"].mean().item()
                )
            else:
                # Single stochastic model
                _, _, dist_params = model(obs, action, deterministic=False)

                all_aleatoric_state.append(
                    dist_params["state_std"].mean().item()
                )
                all_aleatoric_reward.append(
                    dist_params["reward_std"].mean().item()
                )

    results = {
        "mean_aleatoric_state_std": np.mean(all_aleatoric_state),
        "mean_aleatoric_reward_std": np.mean(all_aleatoric_reward),
    }

    if all_epistemic_state:  # Ensemble model
        results.update({
            "mean_epistemic_state_std": np.mean(all_epistemic_state),
            "mean_total_state_std": np.mean(all_total_state),
            "mean_epistemic_reward_std": np.mean(all_epistemic_reward),
            "mean_total_reward_std": np.mean(all_total_reward),
            "epistemic_ratio_state": np.mean(all_epistemic_state) / np.mean(all_total_state),
            "epistemic_ratio_reward": np.mean(all_epistemic_reward) / np.mean(all_total_reward),
        })

    return results


def main():
    """Main training routine for stochastic models."""
    parser = argparse.ArgumentParser(description="Train stochastic world model")

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(config.DATA_PATHS["train_data"]),
        help="Path to training data"
    )
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        help="Train ensemble of stochastic models"
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of models in ensemble"
    )
    parser.add_argument(
        "--bootstrap_ratio",
        type=float,
        default=1.0,
        help="Bootstrap sampling ratio"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay"
    )
    parser.add_argument(
        "--min_std",
        type=float,
        default=0.01,
        help="Minimum standard deviation"
    )
    parser.add_argument(
        "--max_std",
        type=float,
        default=1.0,
        help="Maximum standard deviation"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(config.MODEL_PATHS["world_model"]),
        help="Path to save model"
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    observations, actions, next_observations, rewards = load_data(Path(args.data_path))

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    print(f"Data shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print()

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        observations, actions, next_observations, rewards,
        train_ratio=config.TRAINING_CONFIG["train_ratio"]
    )

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print()

    # Setup device
    device = torch.device(config.DEVICE_CONFIG["device"])
    print(f"Using device: {device}")
    print()

    # Training configuration
    training_config = config.TRAINING_CONFIG["world_model"].copy()
    training_config.update({
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    })

    if args.use_ensemble:
        # Train ensemble
        print(f"Training Stochastic Ensemble with {args.ensemble_size} models")
        print(f"Bootstrap ratio: {args.bootstrap_ratio}")
        print()

        model, histories = train_ensemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            ensemble_size=args.ensemble_size,
            bootstrap_ratio=args.bootstrap_ratio,
            **training_config
        )

        # Save ensemble
        save_path = Path(args.save_path).parent / "stochastic_ensemble.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\nSaved stochastic ensemble to {save_path}")

        # Save training history
        history_path = save_path.parent / "stochastic_ensemble_history.json"
        with open(history_path, "w") as f:
            json.dump(histories, f, indent=2)

    else:
        # Train single model
        print("Training Single Stochastic World Model")
        print()

        model = StochasticWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            min_std=args.min_std,
            max_std=args.max_std,
            **config.MODEL_CONFIG.get("stochastic_world_model", {})
        ).to(device)

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

        # Train
        history = train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            **training_config
        )

        # Save model
        save_path = Path(args.save_path).parent / "stochastic_model.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\nSaved stochastic model to {save_path}")

        # Save training history
        history_path = save_path.parent / "stochastic_model_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # Analyze uncertainty
    print("\n" + "="*60)
    print("Uncertainty Analysis")
    print("="*60)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=config.DEVICE_CONFIG["num_workers"],
    )

    uncertainty_metrics = analyze_uncertainty(model, val_loader, device)

    print("\nUncertainty Decomposition:")
    for key, value in uncertainty_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save uncertainty analysis
    uncertainty_path = Path(args.save_path).parent / "uncertainty_analysis.json"
    with open(uncertainty_path, "w") as f:
        json.dump(uncertainty_metrics, f, indent=2)

    print(f"\nSaved uncertainty analysis to {uncertainty_path}")


if __name__ == "__main__":
    main()