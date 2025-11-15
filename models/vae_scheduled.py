"""
Variational Autoencoder with scheduled beta (KL weight) for improved training.
Implements various scheduling strategies for the KL divergence weight.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable
from models.vae import VAE, VAEEncoder, VAEDecoder


class BetaSchedule:
    """Base class for beta scheduling strategies."""

    def get_beta(self, step: int, total_steps: int) -> float:
        raise NotImplementedError


class ConstantBeta(BetaSchedule):
    """Constant beta throughout training."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def get_beta(self, step: int, total_steps: int) -> float:
        return self.beta


class LinearAnnealingBeta(BetaSchedule):
    """Linear annealing from initial to final beta."""

    def __init__(self, initial: float = 0.0, final: float = 1.0, warmup_steps: int = 1000):
        self.initial = initial
        self.final = final
        self.warmup_steps = warmup_steps

    def get_beta(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.initial + (self.final - self.initial) * (step / self.warmup_steps)
        return self.final


class CosineAnnealingBeta(BetaSchedule):
    """Cosine annealing schedule for beta."""

    def __init__(self, initial: float = 0.0001, final: float = 0.01, warmup_steps: int = 0):
        self.initial = initial
        self.final = final
        self.warmup_steps = warmup_steps

    def get_beta(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.initial

        adjusted_step = step - self.warmup_steps
        adjusted_total = total_steps - self.warmup_steps

        if adjusted_total <= 0:
            return self.final

        cosine_factor = 0.5 * (1 + np.cos(np.pi * adjusted_step / adjusted_total))
        return self.initial + (self.final - self.initial) * (1 - cosine_factor)


class CyclicalBeta(BetaSchedule):
    """Cyclical beta schedule for exploring different regularization strengths."""

    def __init__(self, min_beta: float = 0.0001, max_beta: float = 0.1, cycle_length: int = 1000):
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.cycle_length = cycle_length

    def get_beta(self, step: int, total_steps: int) -> float:
        cycle_position = (step % self.cycle_length) / self.cycle_length
        # Use cosine for smooth transitions
        beta = self.min_beta + (self.max_beta - self.min_beta) * 0.5 * (1 + np.cos(2 * np.pi * cycle_position))
        return beta


class MonotonicBeta(BetaSchedule):
    """Monotonically increasing beta with different growth rates."""

    def __init__(self, initial: float = 0.0001, final: float = 0.01, growth: str = 'exponential'):
        self.initial = initial
        self.final = final
        self.growth = growth

    def get_beta(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return self.final

        progress = step / total_steps

        if self.growth == 'exponential':
            # Exponential growth
            log_range = np.log(self.final / self.initial)
            return self.initial * np.exp(log_range * progress)
        elif self.growth == 'quadratic':
            # Quadratic growth
            return self.initial + (self.final - self.initial) * (progress ** 2)
        else:  # linear
            return self.initial + (self.final - self.initial) * progress


class ScheduledVAE(VAE):
    """VAE with scheduled beta parameter during training."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        encoder_hidden: list = None,
        decoder_hidden: list = None,
        activation: str = "elu",
        beta_schedule: Optional[BetaSchedule] = None,
        free_bits: float = 0.0,  # Minimum KL per dimension
        max_capacity: Optional[float] = None  # Maximum KL capacity
    ):
        # Initialize with beta=1.0, we'll override it during forward pass
        super().__init__(obs_dim, latent_dim, encoder_hidden, decoder_hidden, activation, beta=1.0)

        self.beta_schedule = beta_schedule or ConstantBeta(1.0)
        self.free_bits = free_bits
        self.max_capacity = max_capacity
        self.current_step = 0
        self.total_steps = 1000000  # Default, will be set during training

    def set_training_steps(self, total_steps: int):
        """Set total training steps for proper scheduling."""
        self.total_steps = total_steps

    def forward(self, x: torch.Tensor, step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with scheduled beta.

        Args:
            x: Input observations [batch_size, obs_dim]
            step: Current training step (if None, uses internal counter)

        Returns:
            reconstruction: Reconstructed observations
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            z: Sampled latent representation
        """
        # Get current beta
        if step is None:
            step = self.current_step

        current_beta = self.beta_schedule.get_beta(step, self.total_steps)

        # Update internal beta
        self.beta = current_beta

        # Encode
        mu, log_var = self.encoder(x)

        # Sample latent
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decoder(z)

        # Apply free bits if specified
        if self.free_bits > 0 and self.training:
            # Compute KL per dimension
            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            # Apply free bits threshold
            kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(self.free_bits))
            # Sum over dimensions for total KL
            kl_loss = kl_per_dim.sum(dim=-1).mean()
        else:
            # Standard KL computation
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        # Apply capacity constraint if specified
        if self.max_capacity is not None and self.training:
            kl_loss = torch.minimum(kl_loss, torch.tensor(self.max_capacity))

        # Store KL for loss computation
        self.last_kl_loss = kl_loss

        # Increment step counter
        if self.training:
            self.current_step += 1

        return reconstruction, mu, log_var, z

    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor, step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss with scheduled beta.

        Args:
            x: Original input
            reconstruction: Reconstructed input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            step: Current training step

        Returns:
            total_loss: Combined reconstruction and KL loss
            recon_loss: Reconstruction loss only
            kl_loss: KL divergence loss only
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')

        # Use pre-computed KL if available
        if hasattr(self, 'last_kl_loss'):
            kl_loss = self.last_kl_loss
        else:
            # Compute KL loss
            if self.free_bits > 0:
                kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(self.free_bits))
                kl_loss = kl_per_dim.sum(dim=-1).mean()
            else:
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

            if self.max_capacity is not None:
                kl_loss = torch.minimum(kl_loss, torch.tensor(self.max_capacity))

        # Get current beta
        if step is None:
            step = self.current_step
        current_beta = self.beta_schedule.get_beta(step, self.total_steps)

        # Total loss with scheduled beta
        total_loss = recon_loss + current_beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def get_current_beta(self) -> float:
        """Get the current beta value."""
        return self.beta_schedule.get_beta(self.current_step, self.total_steps)


class CapacityScheduledVAE(ScheduledVAE):
    """VAE with capacity-based scheduling (gradually increases KL capacity)."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        encoder_hidden: list = None,
        decoder_hidden: list = None,
        activation: str = "elu",
        initial_capacity: float = 0.0,
        final_capacity: float = 25.0,
        capacity_steps: int = 25000
    ):
        super().__init__(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            activation=activation,
            beta_schedule=ConstantBeta(1.0)  # Use constant beta
        )

        self.initial_capacity = initial_capacity
        self.final_capacity = final_capacity
        self.capacity_steps = capacity_steps

    def get_capacity(self, step: int) -> float:
        """Get current KL capacity."""
        if step >= self.capacity_steps:
            return self.final_capacity

        progress = step / self.capacity_steps
        return self.initial_capacity + (self.final_capacity - self.initial_capacity) * progress

    def forward(self, x: torch.Tensor, step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with capacity scheduling."""
        if step is None:
            step = self.current_step

        # Update max capacity
        self.max_capacity = self.get_capacity(step)

        return super().forward(x, step)


def create_scheduled_vae(
    obs_dim: int,
    latent_dim: int = 32,
    schedule_type: str = 'cosine',
    initial_beta: float = 0.0001,
    final_beta: float = 0.01,
    warmup_steps: int = 1000,
    **kwargs
) -> ScheduledVAE:
    """
    Factory function to create VAE with specified scheduling.

    Args:
        obs_dim: Observation dimension
        latent_dim: Latent dimension
        schedule_type: Type of scheduling ('constant', 'linear', 'cosine', 'cyclical', 'monotonic')
        initial_beta: Initial beta value
        final_beta: Final beta value
        warmup_steps: Number of warmup steps
        **kwargs: Additional arguments for VAE

    Returns:
        ScheduledVAE instance
    """
    # Create beta schedule
    if schedule_type == 'constant':
        beta_schedule = ConstantBeta(initial_beta)
    elif schedule_type == 'linear':
        beta_schedule = LinearAnnealingBeta(initial_beta, final_beta, warmup_steps)
    elif schedule_type == 'cosine':
        beta_schedule = CosineAnnealingBeta(initial_beta, final_beta, warmup_steps)
    elif schedule_type == 'cyclical':
        beta_schedule = CyclicalBeta(initial_beta, final_beta, cycle_length=warmup_steps)
    elif schedule_type == 'monotonic':
        beta_schedule = MonotonicBeta(initial_beta, final_beta, growth='exponential')
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    return ScheduledVAE(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        beta_schedule=beta_schedule,
        **kwargs
    )


if __name__ == "__main__":
    # Test different scheduling strategies
    import matplotlib.pyplot as plt

    total_steps = 10000
    schedules = {
        'Constant': ConstantBeta(0.001),
        'Linear': LinearAnnealingBeta(0.0001, 0.01, 2000),
        'Cosine': CosineAnnealingBeta(0.0001, 0.01, 1000),
        'Cyclical': CyclicalBeta(0.0001, 0.01, 2000),
        'Monotonic (exp)': MonotonicBeta(0.0001, 0.01, 'exponential'),
        'Monotonic (quad)': MonotonicBeta(0.0001, 0.01, 'quadratic')
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, schedule in schedules.items():
        steps = np.arange(total_steps)
        betas = [schedule.get_beta(step, total_steps) for step in steps]
        ax.plot(steps, betas, label=name, linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Beta (KL Weight)')
    ax.set_title('VAE Beta Scheduling Strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('logs/beta_schedules.png', dpi=150)
    print("Beta scheduling comparison saved to logs/beta_schedules.png")