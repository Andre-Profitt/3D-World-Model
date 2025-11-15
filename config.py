"""
Configuration file for 3D World Model project.

Contains all hyperparameters and settings for the environment, models, and training.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, WEIGHTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Environment configuration
ENV_CONFIG = {
    "world_size": (10.0, 10.0, 10.0),  # (width, depth, height)
    "dt": 0.05,                        # Physics timestep
    "max_velocity": 2.0,               # Maximum velocity magnitude
    "max_acceleration": 5.0,           # Maximum acceleration magnitude
    "goal_radius": 0.5,                # Distance to goal for success
    "seed": 42,                        # Random seed for reproducibility
}

# Visual observation configuration
VISUAL_CONFIG = {
    "obs_type": "state",               # "state", "image", or "both"
    "image_size": (64, 64),           # Image dimensions (H, W)
    "camera_mode": "top_down",        # "top_down", "agent_centric", "fixed_3d"
    "grayscale": True,                # Use grayscale images
    "image_channels": 1,              # 1 for grayscale, 3 for RGB
}

# Data collection
DATA_COLLECTION = {
    "num_episodes": 1000,
    "max_steps_per_episode": 200,
    "random_policy_seed": 123,
    "save_frequency": 100,  # Save data every N episodes
}

# Model architecture
MODEL_CONFIG = {
    # Ensemble settings
    "use_ensemble": False,  # Whether to use ensemble of models
    "ensemble_size": 5,     # Number of models in ensemble
    "bootstrap_ratio": 1.0, # Ratio of data to sample for each member

    # World model
    "world_model": {
        "hidden_dims": [256, 256, 256],
        "activation": "relu",
        "dropout": 0.0,
        "layer_norm": False,
        "predict_delta": True,  # Predict state changes instead of absolute states
    },

    # Encoder/decoder for latent representation
    "encoder": {
        "hidden_dims": [128, 128],
        "latent_dim": 16,  # Compressed representation
        "activation": "relu",
        "layer_norm": True,
        "dropout": 0.0,
    },

    "decoder": {
        "hidden_dims": [128, 128],
        "activation": "relu",
        "layer_norm": True,
        "dropout": 0.0,
    },

    # Latent world model
    "latent_world_model": {
        "hidden_dims": [128, 128],
        "activation": "relu",
        "predict_delta": True,
        "separate_reward_head": True,
        "beta_recon": 0.1,  # Reconstruction loss weight
    },

    # Stochastic world model
    "stochastic_world_model": {
        "hidden_dims": [256, 256, 256],
        "activation": "relu",
        "dropout": 0.0,
        "layer_norm": False,
        "predict_delta": True,
        "min_std": 0.01,
        "max_std": 1.0,
        "separate_reward_head": True,
        "deterministic": False,
    },

    # Stochastic VAE model
    "stochastic_vae": {
        "latent_dim": 32,
        "encoder_hidden": [256, 256],
        "decoder_hidden": [256, 256],
        "dynamics_hidden": [256, 256],
        "activation": "relu",
        "predict_delta": True,
        "beta": 1.0,  # KL divergence weight
        "free_nats": 3.0,  # Minimum KL value
        "reconstruction_loss": "mse",  # "mse" or "gaussian_nll"
    },

    # Vision encoder for image observations
    "vision_encoder": {
        "architecture": "simple",  # "simple" or "resnet"
        "latent_dim": 32,
        "channels": [32, 64, 128],
        "kernel_sizes": [4, 4, 4],
        "strides": [2, 2, 2],
        "batch_norm": True,
        "activation": "relu",
        "dropout": 0.0,
    },
}

# Training configuration
TRAINING_CONFIG = {
    # World model training
    "world_model": {
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "num_epochs": 100,
        "gradient_clip": 1.0,
        "lr_scheduler": {
            "type": "cosine",
            "min_lr": 1e-5,
        },
        "early_stopping": {
            "patience": 10,
            "min_delta": 1e-4,
        },
    },

    # Data split
    "train_ratio": 0.9,
    "val_ratio": 0.1,

    # Logging
    "log_frequency": 10,  # Log every N batches
    "eval_frequency": 1,  # Evaluate every N epochs
    "save_frequency": 5,  # Save checkpoint every N epochs
}

# MPC Controller configuration
MPC_CONFIG = {
    "horizon": 15,                # Planning horizon
    "num_samples": 1024,          # Number of action sequences to sample
    "num_elite": 64,              # Elite samples for CEM (if using)
    "gamma": 0.99,                # Discount factor
    "temperature": 0.1,           # Temperature for action sampling
    "optimization_iters": 3,      # CEM optimization iterations
    "action_noise": 0.3,          # Noise for action sampling
    "use_cem": True,              # Use CEM instead of random shooting
    "lambda_risk": 0.0,           # Risk penalty weight (0=risk-neutral, >0=risk-averse)

    # Risk-aware planning
    "risk_sensitive": {
        "enabled": False,         # Enable risk-sensitive planning
        "num_particles": 10,      # Number of stochastic particles
        "risk_metric": "cvar",    # "cvar", "var", "worst_case", "mean_std", "entropic"
        "risk_level": 0.1,        # Risk level for CVaR/VaR (0.1 = 10% worst cases)
        "lambda_risk": 1.0,       # Risk aversion strength
    },
}

# Evaluation configuration
EVAL_CONFIG = {
    "num_episodes": 50,
    "max_steps": 500,
    "render": True,
    "save_videos": True,
    "video_frequency": 10,  # Save video every N episodes
}

# Experiment tracking
EXPERIMENT_CONFIG = {
    "name": "3d_world_model_baseline",
    "tags": ["3d", "world_model", "mpc"],
    "notes": "Baseline 3D world model with MPC control",
    "use_tensorboard": True,
    "use_wandb": False,  # Optional: Weights & Biases tracking
}

# Device configuration
DEVICE_CONFIG = {
    "device": "cuda" if os.environ.get("USE_CUDA") else "mps" if os.environ.get("USE_MPS") else "cpu",
    "num_workers": 0,  # Set to 0 to avoid multiprocessing issues
    "pin_memory": False,  # Disabled for MPS
}

# File paths
MODEL_PATHS = {
    "world_model": WEIGHTS_DIR / "world_model.pt",
    "encoder": WEIGHTS_DIR / "encoder.pt",
    "decoder": WEIGHTS_DIR / "decoder.pt",
    "autoencoder": WEIGHTS_DIR / "autoencoder.pt",
    "latent_world_model": WEIGHTS_DIR / "latent_world_model.pt",
    "optimizer": WEIGHTS_DIR / "optimizer.pt",
    "best_model": WEIGHTS_DIR / "best_world_model.pt",
    "best_autoencoder": WEIGHTS_DIR / "best_autoencoder.pt",
    "best_latent_model": WEIGHTS_DIR / "best_latent_world_model.pt",
}

DATA_PATHS = {
    "train_data": DATA_DIR / "train_data.npz",
    "val_data": DATA_DIR / "val_data.npz",
    "raw_trajectories": DATA_DIR / "raw_trajectories.npz",
    "visual_data": DATA_DIR / "visual_data.npz",
    "train_visual_data": DATA_DIR / "train_visual_data.npz",
    "val_visual_data": DATA_DIR / "val_visual_data.npz",
}

# Reproducibility
SEED = 42

def get_config(config_name: str = None):
    """
    Get configuration dictionary by name.

    Args:
        config_name: Name of configuration to retrieve

    Returns:
        Configuration dictionary
    """
    configs = {
        "env": ENV_CONFIG,
        "data": DATA_COLLECTION,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "mpc": MPC_CONFIG,
        "eval": EVAL_CONFIG,
        "experiment": EXPERIMENT_CONFIG,
        "device": DEVICE_CONFIG,
        "paths": {
            "model": MODEL_PATHS,
            "data": DATA_PATHS,
        },
    }

    if config_name:
        return configs.get(config_name, {})
    return configs