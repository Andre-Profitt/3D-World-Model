#!/usr/bin/env python3
"""
Curriculum Learning for Latent Dynamics Model.
Progressively increases prediction horizon during training for better long-term predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import Tuple, Dict
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.vae import VAE
from models.simple_latent_dynamics import SimpleLatentDynamics


class CurriculumSchedule:
    """Manages curriculum progression for prediction horizons."""

    def __init__(
        self,
        initial_horizon: int = 5,
        final_horizon: int = 50,
        stages: list = None,
        patience: int = 10,
        improvement_threshold: float = 0.01
    ):
        self.stages = stages or [5, 10, 15, 20, 25, 30, 40, 50]
        self.current_stage = 0
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.stage_epochs = []

    def get_current_horizon(self) -> int:
        """Get current training horizon."""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]

    def should_advance(self, val_loss: float, epoch: int) -> bool:
        """Check if we should advance to next stage."""
        # Check if we've improved enough
        improvement = (self.best_loss - val_loss) / (self.best_loss + 1e-8)

        if improvement > self.improvement_threshold:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Advance if we've been patient enough and performance is stable
        if self.patience_counter >= self.patience and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.patience_counter = 0
            self.best_loss = float('inf')
            self.stage_epochs.append(epoch)
            return True

        return False

    def get_progress(self) -> Dict:
        """Get curriculum progress info."""
        return {
            'current_stage': self.current_stage,
            'current_horizon': self.get_current_horizon(),
            'max_horizon': self.stages[-1],
            'progress': (self.current_stage + 1) / len(self.stages),
            'stage_epochs': self.stage_epochs
        }


class MultiStepLatentDataset(Dataset):
    """Dataset for multi-step latent predictions with variable horizon."""

    def __init__(self, latent_states, actions, next_states, rewards, max_horizon=50):
        self.latent_states = latent_states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.max_horizon = max_horizon

    def __len__(self):
        return len(self.latent_states) - self.max_horizon

    def __getitem__(self, idx):
        # Return sequence starting at idx
        horizon = min(self.max_horizon, len(self.latent_states) - idx - 1)

        states = self.latent_states[idx:idx+horizon]
        actions = self.actions[idx:idx+horizon]
        targets = self.next_states[idx:idx+horizon]
        rewards = self.rewards[idx:idx+horizon] if self.rewards is not None else None

        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'targets': torch.FloatTensor(targets),
            'rewards': torch.FloatTensor(rewards) if rewards is not None else torch.zeros(horizon),
            'horizon': horizon
        }


def multi_step_loss(
    model: nn.Module,
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    horizon: int,
    gamma: float = 0.95,
    normalize_by_horizon: bool = True
) -> torch.Tensor:
    """
    Compute multi-step prediction loss with exponential weighting.

    Args:
        model: Dynamics model
        initial_state: Initial latent state [B, latent_dim]
        actions: Action sequence [B, H, action_dim]
        targets: Target state sequence [B, H, latent_dim]
        horizon: Prediction horizon
        gamma: Exponential decay factor for weighting
        normalize_by_horizon: Whether to normalize loss by horizon

    Returns:
        Weighted multi-step loss
    """
    batch_size = initial_state.size(0)
    total_loss = 0.0
    weight_sum = 0.0

    state = initial_state
    state_losses = []

    for t in range(horizon):
        # Predict next state
        if t < actions.size(1):  # Check if we have actions for this timestep
            action = actions[:, t]
            next_state_pred, reward_pred = model(state, action)

            # Compute loss for this timestep
            if t < targets.size(1):  # Check if we have targets for this timestep
                target = targets[:, t]
                step_loss = nn.functional.mse_loss(next_state_pred, target, reduction='mean')

                # Apply exponential weighting
                weight = gamma ** t
                total_loss += weight * step_loss
                weight_sum += weight

                state_losses.append(step_loss.item())

            # Update state for next prediction
            state = next_state_pred

    # Normalize by weight sum
    if weight_sum > 0:
        total_loss = total_loss / weight_sum

    # Optionally normalize by horizon to make losses comparable across stages
    if normalize_by_horizon and horizon > 0:
        total_loss = total_loss * (1 - gamma) / (1 - gamma ** horizon)

    return total_loss


def train_with_curriculum(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    curriculum: CurriculumSchedule,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    gamma: float = 0.95,
    device: str = 'cpu'
) -> Dict:
    """Train dynamics model with curriculum learning."""

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {
        'train_losses': [],
        'val_losses': [],
        'horizons': [],
        'stage_transitions': []
    }

    print("\n" + "="*60)
    print("CURRICULUM LEARNING FOR DYNAMICS MODEL")
    print("="*60)
    print(f"Initial horizon: {curriculum.get_current_horizon()}")
    print(f"Target horizon: {curriculum.stages[-1]}")
    print(f"Stages: {curriculum.stages}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    for epoch in range(epochs):
        # Get current training horizon
        current_horizon = curriculum.get_current_horizon()

        # Training
        model.train()
        train_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [H={current_horizon}]")
        for batch in progress_bar:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            targets = batch['targets'].to(device)

            # Use current horizon for training
            batch_horizon = min(current_horizon, states.size(1))

            if batch_horizon > 0:
                # Get initial state
                initial_state = states[:, 0] if states.dim() > 1 else states

                # Truncate sequences to current horizon
                actions_truncated = actions[:, :batch_horizon]
                targets_truncated = targets[:, :batch_horizon]

                # Compute multi-step loss
                loss = multi_step_loss(
                    model, initial_state, actions_truncated,
                    targets_truncated, batch_horizon, gamma
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                targets = batch['targets'].to(device)

                batch_horizon = min(current_horizon, states.size(1))

                if batch_horizon > 0:
                    initial_state = states[:, 0] if states.dim() > 1 else states
                    actions_truncated = actions[:, :batch_horizon]
                    targets_truncated = targets[:, :batch_horizon]

                    loss = multi_step_loss(
                        model, initial_state, actions_truncated,
                        targets_truncated, batch_horizon, gamma
                    )

                    val_losses.append(loss.item())

        # Calculate epoch losses
        train_loss = np.mean(train_losses) if train_losses else float('inf')
        val_loss = np.mean(val_losses) if val_losses else float('inf')

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['horizons'].append(current_horizon)

        # Check if we should advance curriculum
        if curriculum.should_advance(val_loss, epoch):
            new_horizon = curriculum.get_current_horizon()
            print(f"\n🎯 Advancing curriculum: {current_horizon} → {new_horizon} steps")
            history['stage_transitions'].append(epoch)

        # Update learning rate
        scheduler.step()

        # Print epoch summary
        if (epoch + 1) % 10 == 0:
            progress = curriculum.get_progress()
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Horizon: {current_horizon} (Stage {progress['current_stage']+1}/{len(curriculum.stages)})")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Progress: {progress['progress']*100:.1f}%")

    return history


def plot_curriculum_training(history: Dict, save_path: str = 'logs/curriculum_training.png'):
    """Plot curriculum training progress."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot losses
    ax = axes[0]
    epochs = np.arange(len(history['train_losses']))
    ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', alpha=0.7)
    ax.plot(epochs, history['val_losses'], 'r-', label='Val Loss', alpha=0.7)

    # Mark stage transitions
    for transition_epoch in history['stage_transitions']:
        ax.axvline(x=transition_epoch, color='g', linestyle='--', alpha=0.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Curriculum Learning Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot horizon progression
    ax = axes[1]
    ax.plot(epochs, history['horizons'], 'g-', linewidth=2)
    ax.fill_between(epochs, 0, history['horizons'], alpha=0.3, color='g')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Horizon')
    ax.set_title('Curriculum Horizon Progression')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCurriculum training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train dynamics with curriculum learning')
    parser.add_argument('--vae_path', type=str, default='weights/best_vae.pt',
                       help='Path to trained VAE model')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor for multi-step loss')
    parser.add_argument('--initial_horizon', type=int, default=5,
                       help='Initial prediction horizon')
    parser.add_argument('--final_horizon', type=int, default=50,
                       help='Final prediction horizon')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for dynamics model')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of layers in dynamics model')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VAE for encoding
    print("Loading VAE model...")
    vae_checkpoint = torch.load(args.vae_path, map_location=device)
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
    print("Loading training data...")
    data = np.load('data/raw_trajectories.npz')
    observations = data['observations']
    actions = data['actions']
    next_observations = data['next_observations']
    rewards = data['rewards'] if 'rewards' in data else None

    # Load normalization stats
    with open('weights/normalization_stats.json', 'r') as f:
        norm_stats = json.load(f)
    mean = np.array(norm_stats['mean'])
    std = np.array(norm_stats['std'])

    # Normalize and encode observations
    print("Encoding observations to latent space...")
    observations_norm = (observations - mean) / (std + 1e-8)
    next_observations_norm = (next_observations - mean) / (std + 1e-8)

    with torch.no_grad():
        latent_states = vae.encode(torch.FloatTensor(observations_norm)).numpy()
        next_latent_states = vae.encode(torch.FloatTensor(next_observations_norm)).numpy()

    # Split data
    n_train = int(0.9 * len(latent_states))

    train_dataset = MultiStepLatentDataset(
        latent_states[:n_train],
        actions[:n_train],
        next_latent_states[:n_train],
        rewards[:n_train] if rewards is not None else None,
        max_horizon=args.final_horizon
    )

    val_dataset = MultiStepLatentDataset(
        latent_states[n_train:],
        actions[n_train:],
        next_latent_states[n_train:],
        rewards[n_train:] if rewards is not None else None,
        max_horizon=args.final_horizon
    )

    # Create dynamics model
    latent_dim = vae_config.get('latent_dim', 24)
    action_dim = actions.shape[-1]

    model = SimpleLatentDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    )

    # Create curriculum schedule
    curriculum = CurriculumSchedule(
        initial_horizon=args.initial_horizon,
        final_horizon=args.final_horizon,
        stages=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        patience=15,
        improvement_threshold=0.01
    )

    # Train with curriculum
    history = train_with_curriculum(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        curriculum=curriculum,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=device
    )

    # Save model
    save_path = 'weights/curriculum_dynamics_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'action_dim': action_dim,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'curriculum_history': history,
        'final_horizon': curriculum.get_current_horizon()
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # Plot training progress
    plot_curriculum_training(history)

    # Print final summary
    print("\n" + "="*60)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*60)
    print(f"Final horizon reached: {curriculum.get_current_horizon()} steps")
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    print(f"Final val loss: {history['val_losses'][-1]:.6f}")
    print(f"Stage transitions at epochs: {history['stage_transitions']}")
    print("="*60)


if __name__ == "__main__":
    main()