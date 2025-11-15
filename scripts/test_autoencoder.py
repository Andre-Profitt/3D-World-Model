"""
Quick test of autoencoder reconstruction quality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models import Encoder, Decoder
import config

# Load encoder and decoder
obs_dim = 9
latent_dim = config.MODEL_CONFIG["encoder"]["latent_dim"]

print(f"Loading autoencoder (obs: {obs_dim}D, latent: {latent_dim}D)...")

# Load encoder
encoder = Encoder(
    obs_dim=obs_dim,
    latent_dim=latent_dim,
    **{k: v for k, v in config.MODEL_CONFIG["encoder"].items() if k != "latent_dim"}
)

encoder_checkpoint = torch.load(
    config.WEIGHTS_DIR / "encoder.pt",
    map_location="cpu"
)
encoder.load_state_dict(encoder_checkpoint["model_state_dict"])
encoder.eval()
print(f"Loaded encoder from epoch {encoder_checkpoint.get('epoch', 'unknown')}")

# Load decoder
decoder = Decoder(
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    **{k: v for k, v in config.MODEL_CONFIG["decoder"].items() if k != "latent_dim"}
)

decoder_checkpoint = torch.load(
    config.WEIGHTS_DIR / "decoder.pt",
    map_location="cpu"
)
decoder.load_state_dict(decoder_checkpoint["model_state_dict"])
decoder.eval()
print(f"Loaded decoder from epoch {decoder_checkpoint.get('epoch', 'unknown')}")

# Test on random observations
print("\nTesting reconstruction on random observations...")
test_obs = torch.randn(10, obs_dim)

with torch.no_grad():
    # Encode
    latent = encoder(test_obs)
    print(f"Encoded shape: {latent.shape}")

    # Decode
    reconstructed = decoder(latent)
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Compute error
    recon_error = torch.mean((test_obs - reconstructed) ** 2).item()
    print(f"MSE reconstruction error: {recon_error:.6f}")

    # L2 error
    l2_error = torch.mean(torch.norm(test_obs - reconstructed, dim=-1)).item()
    print(f"L2 reconstruction error: {l2_error:.6f}")

# Test on typical environment observations
print("\nTesting on typical environment observations...")
# Position (0-10), velocity (-5 to 5), goal (0-10)
typical_obs = torch.zeros(5, obs_dim)
typical_obs[:, :3] = torch.rand(5, 3) * 10  # position
typical_obs[:, 3:6] = (torch.rand(5, 3) - 0.5) * 10  # velocity
typical_obs[:, 6:9] = torch.rand(5, 3) * 10  # goal

with torch.no_grad():
    latent = encoder(typical_obs)
    reconstructed = decoder(latent)

    mse_error = torch.mean((typical_obs - reconstructed) ** 2).item()
    l2_error = torch.mean(torch.norm(typical_obs - reconstructed, dim=-1)).item()

    print(f"Typical observation MSE: {mse_error:.6f}")
    print(f"Typical observation L2: {l2_error:.6f}")

    # Show example
    print("\nExample reconstruction:")
    print(f"Original:      {typical_obs[0].numpy()}")
    print(f"Reconstructed: {reconstructed[0].numpy()}")
    print(f"Difference:    {(typical_obs[0] - reconstructed[0]).numpy()}")

print("\nDone!")