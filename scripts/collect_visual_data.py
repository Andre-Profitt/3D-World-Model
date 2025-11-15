"""
Collect visual observation data for training vision-based world models.

Collects both state and image observations for multi-modal learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from envs import Simple3DNavEnv
import config


def collect_visual_episode(
    env: Simple3DNavEnv,
    policy: str = "random",
    max_steps: int = 200,
    render_freq: int = 0,
) -> dict:
    """
    Collect one episode with visual observations.

    Args:
        env: Environment instance
        policy: Policy to use ("random" or "heuristic")
        max_steps: Maximum steps per episode
        render_freq: Render every N steps (0 = no rendering)

    Returns:
        Episode data including images
    """
    obs = env.reset()

    # Storage
    states = []
    images = []
    actions = []
    next_states = []
    next_images = []
    rewards = []
    dones = []

    for step in range(max_steps):
        # Store current observation
        if env.obs_type == "both":
            state = obs["state"]
            image = obs["image"]
        elif env.obs_type == "image":
            state = np.concatenate([env.state, env.goal])  # Get underlying state
            image = obs
        else:
            state = obs
            image = env._get_image_obs()  # Render image even in state mode

        states.append(state)
        images.append(image)

        # Select action based on policy
        if policy == "random":
            action = env.rng.uniform(
                -env.max_acceleration,
                env.max_acceleration,
                size=3
            )
        elif policy == "heuristic":
            # Simple goal-directed policy
            position = env.state[:3]
            velocity = env.state[3:6]
            goal = env.goal

            direction = goal - position
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction = direction / distance

            desired_velocity = direction * min(distance * 0.5, env.max_velocity)
            acceleration = (desired_velocity - velocity) * 2.0

            action = np.clip(
                acceleration,
                -env.max_acceleration,
                env.max_acceleration
            )
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Step environment
        next_obs, reward, done, info = env.step(action)

        # Store next observation
        if env.obs_type == "both":
            next_state = next_obs["state"]
            next_image = next_obs["image"]
        elif env.obs_type == "image":
            next_state = np.concatenate([env.state, env.goal])
            next_image = next_obs
        else:
            next_state = next_obs
            next_image = env._get_image_obs()

        actions.append(action)
        next_states.append(next_state)
        next_images.append(next_image)
        rewards.append(reward)
        dones.append(done)

        # Render if requested
        if render_freq > 0 and step % render_freq == 0:
            visualize_observation(image, state, action, reward)

        if done:
            break

    return {
        "states": np.array(states),
        "images": np.array(images),
        "actions": np.array(actions),
        "next_states": np.array(next_states),
        "next_images": np.array(next_images),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
    }


def visualize_observation(
    image: np.ndarray,
    state: np.ndarray,
    action: np.ndarray = None,
    reward: float = None,
):
    """Visualize an observation for debugging."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show image
    if image.shape[0] == 1:  # Grayscale
        axes[0].imshow(image[0], cmap='gray')
    else:  # RGB
        axes[0].imshow(image.transpose(1, 2, 0))
    axes[0].set_title("Visual Observation")
    axes[0].axis('off')

    # Show state info
    position = state[:3]
    velocity = state[3:6]
    goal = state[6:9] if len(state) >= 9 else None

    info_text = f"Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]\n"
    info_text += f"Velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]\n"
    if goal is not None:
        dist_to_goal = np.linalg.norm(goal - position)
        info_text += f"Goal: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]\n"
        info_text += f"Distance: {dist_to_goal:.2f}\n"
    if action is not None:
        info_text += f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]\n"
    if reward is not None:
        info_text += f"Reward: {reward:.3f}"

    axes[1].text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center')
    axes[1].set_title("State Information")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.pause(0.01)


def save_visual_dataset(episodes: list, save_path: Path):
    """
    Save collected visual episodes to disk.

    Args:
        episodes: List of episode dictionaries
        save_path: Path to save the dataset
    """
    # Combine all episodes
    all_states = []
    all_images = []
    all_actions = []
    all_next_states = []
    all_next_images = []
    all_rewards = []
    all_dones = []

    for episode in episodes:
        all_states.append(episode["states"])
        all_images.append(episode["images"])
        all_actions.append(episode["actions"])
        all_next_states.append(episode["next_states"])
        all_next_images.append(episode["next_images"])
        all_rewards.append(episode["rewards"])
        all_dones.append(episode["dones"])

    # Concatenate
    dataset = {
        "observations": np.concatenate(all_states, axis=0),
        "image_observations": np.concatenate(all_images, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "next_observations": np.concatenate(all_next_states, axis=0),
        "next_image_observations": np.concatenate(all_next_images, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
        "dones": np.concatenate(all_dones, axis=0),
    }

    # Save
    np.savez_compressed(save_path, **dataset)
    print(f"Saved visual dataset with {len(dataset['observations'])} transitions to {save_path}")

    # Print statistics
    print(f"  State shape: {dataset['observations'].shape}")
    print(f"  Image shape: {dataset['image_observations'].shape}")
    print(f"  Image range: [{dataset['image_observations'].min():.3f}, {dataset['image_observations'].max():.3f}]")
    print(f"  Total reward: {dataset['rewards'].sum():.1f}")
    print(f"  Success rate: {dataset['dones'].mean():.1%}")


def visualize_dataset_samples(dataset_path: Path, num_samples: int = 5):
    """Visualize random samples from the dataset."""
    data = np.load(dataset_path)

    states = data["observations"]
    images = data["image_observations"]
    actions = data["actions"]
    rewards = data["rewards"]

    # Random samples
    indices = np.random.choice(len(states), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Show image
        image = images[idx]
        if image.shape[0] == 1:  # Grayscale
            axes[i, 0].imshow(image[0], cmap='gray')
        else:  # RGB
            axes[i, 0].imshow(image.transpose(1, 2, 0))
        axes[i, 0].set_title(f"Sample {idx}: Image")
        axes[i, 0].axis('off')

        # Show state
        state = states[idx]
        position = state[:3]
        velocity = state[3:6]
        goal = state[6:9] if len(state) >= 9 else None

        axes[i, 1].bar(range(len(state)), state)
        axes[i, 1].set_title(f"Sample {idx}: State (r={rewards[idx]:.2f})")
        axes[i, 1].set_xlabel("Dimension")
        axes[i, 1].set_ylabel("Value")

    plt.tight_layout()
    save_path = config.LOGS_DIR / "visual_dataset_samples.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    """Main data collection routine."""
    parser = argparse.ArgumentParser(description="Collect visual observation data")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "heuristic"],
        default="random",
        help="Policy for action selection"
    )
    parser.add_argument(
        "--camera_mode",
        type=str,
        choices=["top_down", "agent_centric", "fixed_3d"],
        default="top_down",
        help="Camera perspective"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Image size (height width)"
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use grayscale images"
    )
    parser.add_argument(
        "--render_samples",
        type=int,
        default=0,
        help="Render N sample episodes"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(config.DATA_DIR / "visual_data.npz"),
        help="Path to save dataset"
    )
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=5,
        help="Visualize N samples after collection"
    )

    args = parser.parse_args()

    # Create environment with visual observations
    env = Simple3DNavEnv(
        **config.ENV_CONFIG,
        obs_type="both",  # Collect both state and image
        image_size=tuple(args.image_size),
        camera_mode=args.camera_mode,
        grayscale=args.grayscale,
    )

    print("=" * 60)
    print("Visual Data Collection")
    print("=" * 60)
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Policy: {args.policy}")
    print(f"Camera: {args.camera_mode}")
    print(f"Image size: {args.image_size}")
    print(f"Grayscale: {args.grayscale}")
    print()

    # Collect episodes
    episodes = []
    total_steps = 0
    total_reward = 0
    successes = 0

    for episode_idx in tqdm(range(args.num_episodes), desc="Collecting episodes"):
        # Render some episodes
        render_freq = 50 if episode_idx < args.render_samples else 0

        episode_data = collect_visual_episode(
            env,
            policy=args.policy,
            max_steps=args.max_steps,
            render_freq=render_freq,
        )

        episodes.append(episode_data)
        total_steps += len(episode_data["rewards"])
        total_reward += episode_data["rewards"].sum()
        successes += episode_data["dones"][-1] if len(episode_data["dones"]) > 0 else 0

    # Print statistics
    print(f"\nCollection complete!")
    print(f"  Total steps: {total_steps}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Success rate: {successes / args.num_episodes:.1%}")
    print(f"  Avg steps/episode: {total_steps / args.num_episodes:.1f}")
    print(f"  Avg reward/episode: {total_reward / args.num_episodes:.1f}")

    # Save dataset
    save_path = Path(args.save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_visual_dataset(episodes, save_path)

    # Split into train/val
    num_train = int(0.9 * len(episodes))
    train_episodes = episodes[:num_train]
    val_episodes = episodes[num_train:]

    if len(val_episodes) > 0:
        train_path = save_path.parent / f"train_{save_path.name}"
        val_path = save_path.parent / f"val_{save_path.name}"

        save_visual_dataset(train_episodes, train_path)
        save_visual_dataset(val_episodes, val_path)

    # Visualize samples
    if args.visualize_samples > 0:
        print(f"\nVisualizing {args.visualize_samples} samples...")
        visualize_dataset_samples(save_path, args.visualize_samples)


if __name__ == "__main__":
    main()