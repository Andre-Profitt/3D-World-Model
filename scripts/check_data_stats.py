"""
Check statistics of training data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config

# Load training data
data_path = config.DATA_DIR / "train_data.npz"
print(f"Loading data from {data_path}")

data = np.load(data_path)
print(f"Available keys: {list(data.keys())}")

# Check which format the data is in
if "observations" in data:
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_observations = data["next_observations"]
else:
    # Might be in trajectory format
    trajectories = data["trajectories"]
    print(f"Data is in trajectory format: {trajectories.shape}")
    # Extract transitions from trajectories
    observations = trajectories[:, :-1, :9].reshape(-1, 9)
    actions = data.get("actions", np.zeros((observations.shape[0], 3)))
    rewards = data.get("rewards", np.zeros(observations.shape[0]))
    next_observations = trajectories[:, 1:, :9].reshape(-1, 9)

print(f"\nData shapes:")
print(f"  Observations: {observations.shape}")
print(f"  Actions: {actions.shape}")
print(f"  Rewards: {rewards.shape}")
print(f"  Next observations: {next_observations.shape}")

print(f"\nObservation statistics:")
print(f"  Mean: {observations.mean(axis=0)}")
print(f"  Std:  {observations.std(axis=0)}")
print(f"  Min:  {observations.min(axis=0)}")
print(f"  Max:  {observations.max(axis=0)}")

print(f"\nPer-component ranges:")
components = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "goal_x", "goal_y", "goal_z"]
for i, name in enumerate(components):
    print(f"  {name:8}: [{observations[:, i].min():.3f}, {observations[:, i].max():.3f}] "
          f"(mean: {observations[:, i].mean():.3f}, std: {observations[:, i].std():.3f})")

print(f"\nAction statistics:")
print(f"  Mean: {actions.mean(axis=0)}")
print(f"  Std:  {actions.std(axis=0)}")
print(f"  Min:  {actions.min():.3f}")
print(f"  Max:  {actions.max():.3f}")

print(f"\nReward statistics:")
print(f"  Mean: {rewards.mean():.3f}")
print(f"  Std:  {rewards.std():.3f}")
print(f"  Min:  {rewards.min():.3f}")
print(f"  Max:  {rewards.max():.3f}")