import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.vision_encoder import create_visual_encoder_decoder_pair
from models.latent_world_model import LatentWorldModel
from envs.simple_3d_nav import Simple3DNavEnv
import wm_config as config

class VisualTransitionDataset(Dataset):
    def __init__(self, observations, actions, next_observations, rewards):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)
        self.next_observations = torch.FloatTensor(next_observations)
        self.rewards = torch.FloatTensor(rewards)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.next_observations[idx],
            self.rewards[idx]
        )

def collect_visual_data(num_episodes=100, max_steps=200, save_path=None):
    """Collect visual data from the environment."""
    cfg = config.get_config()
    env_config = cfg["env"]
    visual_config = cfg.get("visual", config.VISUAL_CONFIG) # Fallback if not in get_config
    
    print("Initializing environment for data collection...")
    env = Simple3DNavEnv(
        world_size=env_config["world_size"],
        dt=env_config["dt"],
        obs_type="image", # Force image observation
        image_size=visual_config["image_size"],
        camera_mode=visual_config["camera_mode"],
        grayscale=visual_config["grayscale"],
        seed=env_config["seed"]
    )
    
    observations = []
    actions = []
    next_observations = []
    rewards = []
    
    print(f"Collecting {num_episodes} episodes...")
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = env.rng.uniform(-1, 1, size=env.action_dim)
            next_obs, reward, done, _ = env.step(action)
            
            observations.append(obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            
            obs = next_obs
            steps += 1
            
    # Convert to numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)
    next_observations = np.array(next_observations)
    rewards = np.array(rewards)
    
    print(f"Collected {len(observations)} transitions.")
    print(f"Observation shape: {observations.shape}")
    
    if save_path:
        print(f"Saving data to {save_path}...")
        np.savez(
            save_path,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards
        )
        
    return observations, actions, next_observations, rewards

def train_visual_autoencoder(
    encoder, decoder, dataloader, num_epochs=50, device="cpu", save_dir="weights"
):
    """Train the visual autoencoder."""
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    
    encoder.to(device)
    decoder.to(device)
    
    print("Starting Autoencoder Training...")
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for obs, _, _, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            obs = obs.to(device)
            
            # Forward pass
            z = encoder(obs)
            recon = decoder(z)
            
            loss = criterion(recon, obs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(encoder.state_dict(), os.path.join(save_dir, "visual_encoder.pt"))
            torch.save(decoder.state_dict(), os.path.join(save_dir, "visual_decoder.pt"))
            
    # Plot loss
    plt.figure()
    plt.plot(train_losses)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(os.path.join(config.LOGS_DIR, "ae_training_loss.png"))
    plt.close()
    
    return encoder, decoder

def train_latent_dynamics(
    encoder, latent_model, dataloader, num_epochs=50, device="cpu", save_dir="weights"
):
    """Train the latent world model."""
    optimizer = optim.Adam(latent_model.parameters(), lr=1e-3)
    criterion_state = nn.MSELoss()
    criterion_reward = nn.MSELoss()
    
    encoder.eval() # Encoder is fixed
    latent_model.to(device)
    
    print("Starting Latent Dynamics Training...")
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for obs, action, next_obs, reward in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            obs = obs.to(device)
            action = action.to(device)
            next_obs = next_obs.to(device)
            reward = reward.to(device).unsqueeze(1) # (B, 1)
            
            with torch.no_grad():
                z = encoder(obs)
                z_next_target = encoder(next_obs)
                
            # Forward pass
            z_next_pred, reward_pred = latent_model(z, action)
            
            # Loss
            if latent_model.predict_delta:
                delta_pred = z_next_pred
                delta_target = z_next_target - z
                loss_state = criterion_state(delta_pred, delta_target)
            else:
                loss_state = criterion_state(z_next_pred, z_next_target)
                
            loss_reward = criterion_reward(reward_pred, reward)
            
            loss = loss_state + loss_reward
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
             torch.save(latent_model.state_dict(), os.path.join(save_dir, "visual_latent_model.pt"))

    # Plot loss
    plt.figure()
    plt.plot(train_losses)
    plt.title("Latent Dynamics Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(config.LOGS_DIR, "dynamics_training_loss.png"))
    plt.close()
    
    return latent_model

def main():
    # Configuration
    cfg = config.get_config()
    device = cfg["device"]["device"]
    visual_config = cfg.get("visual", config.VISUAL_CONFIG)
    
    # 1. Collect Data
    data_path = config.DATA_PATHS["train_visual_data"]
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}")
        data = np.load(data_path)
        observations = data["observations"]
        actions = data["actions"]
        next_observations = data["next_observations"]
        rewards = data["rewards"]
    else:
        observations, actions, next_observations, rewards = collect_visual_data(
            num_episodes=100, # Reduced for quick testing
            save_path=data_path
        )
        
    dataset = VisualTransitionDataset(observations, actions, next_observations, rewards)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Train Autoencoder
    print("Initializing Autoencoder...")
    latent_dim = 32
    input_channels = observations.shape[1] # C, H, W
    
    encoder, decoder = create_visual_encoder_decoder_pair(
        latent_dim=latent_dim,
        input_channels=input_channels,
        image_size=visual_config["image_size"][0]
    )
    
    encoder, decoder = train_visual_autoencoder(
        encoder, decoder, dataloader, num_epochs=20, device=device, save_dir=config.WEIGHTS_DIR
    )
    
    # 3. Train Latent Dynamics
    print("Initializing Latent World Model...")
    latent_model = LatentWorldModel(
        latent_dim=latent_dim,
        action_dim=3, # From env
        hidden_dims=[128, 128],
        predict_delta=True
    )
    
    latent_model = train_latent_dynamics(
        encoder, latent_model, dataloader, num_epochs=30, device=device, save_dir=config.WEIGHTS_DIR
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
