"""
Data collection script for 3D World Model.

Collects experience data using random or heuristic policies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from envs import Simple3DNavEnv
import wm_config as config


class DataCollector:
    """Collects experience data from the environment."""

    def __init__(
        self,
        env: Simple3DNavEnv,
        policy: str = "random",
        seed: int = 42
    ):
        """
        Initialize data collector.

        Args:
            env: Environment instance
            policy: Policy type ("random", "heuristic")
            seed: Random seed
        """
        self.env = env
        self.policy = policy
        self.rng = np.random.default_rng(seed)

        # Data buffers
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.dones = []

    def collect_episode(self, max_steps: int = 200) -> dict:
        """
        Collect data from one episode.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            Episode statistics
        """
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            # Select action based on policy
            action = self._select_action(obs)

            # Execute action
            next_obs, reward, done, info = self.env.step(action)

            # Store transition
            self.observations.append(obs)
            self.actions.append(action)
            self.next_observations.append(next_obs)
            self.rewards.append(reward)
            self.dones.append(done)

            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1

            if done:
                break

        return {
            "reward": episode_reward,
            "length": episode_length,
            "success": info.get("success", False),
            "final_dist": info.get("dist_to_goal", -1),
        }

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Select action based on policy.

        Args:
            obs: Current observation

        Returns:
            Action to execute
        """
        if self.policy == "random":
            # Random acceleration commands
            return self.rng.uniform(
                -self.env.max_acceleration,
                self.env.max_acceleration,
                size=3
            )

        elif self.policy == "heuristic":
            # Simple heuristic: accelerate towards goal
            state = obs[:6]
            goal = obs[6:9]

            position = state[:3]
            velocity = state[3:6]

            # Direction to goal
            direction = goal - position
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction = direction / distance

            # Desired velocity (proportional to distance)
            desired_velocity = direction * min(distance * 0.5, self.env.max_velocity)

            # Acceleration to achieve desired velocity
            acceleration = (desired_velocity - velocity) * 2.0

            # Clip to maximum acceleration
            return np.clip(
                acceleration,
                -self.env.max_acceleration,
                self.env.max_acceleration
            )

        else:
            raise ValueError(f"Unknown policy: {self.policy}")

    def get_data(self) -> dict:
        """
        Get collected data as numpy arrays.

        Returns:
            Dictionary of data arrays
        """
        return {
            "observations": np.array(self.observations, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "next_observations": np.array(self.next_observations, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=bool),
        }

    def save_data(self, filepath: Path):
        """Save collected data to file."""
        data = self.get_data()
        np.savez_compressed(filepath, **data)
        print(f"Saved {len(self.observations)} transitions to {filepath}")


def main():
    """Main data collection routine."""
    parser = argparse.ArgumentParser(description="Collect data for 3D world model")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=config.DATA_COLLECTION["num_episodes"],
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=config.DATA_COLLECTION["max_steps_per_episode"],
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "heuristic"],
        default="random",
        help="Policy to use for data collection"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DATA_COLLECTION["random_policy_seed"],
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.DATA_PATHS["raw_trajectories"]),
        help="Output file path"
    )
    parser.add_argument(
        "--render_samples",
        type=int,
        default=0,
        help="Number of episodes to render"
    )

    args = parser.parse_args()

    # Create environment
    env = Simple3DNavEnv(**config.ENV_CONFIG)

    # Create data collector
    collector = DataCollector(env, policy=args.policy, seed=args.seed)

    # Collect episodes
    print(f"Collecting {args.num_episodes} episodes with {args.policy} policy...")

    episode_stats = []
    render_counter = 0

    for episode in tqdm(range(args.num_episodes)):
        stats = collector.collect_episode(max_steps=args.max_steps)
        episode_stats.append(stats)

        # Optionally render some episodes
        if render_counter < args.render_samples:
            if episode % max(1, args.num_episodes // args.render_samples) == 0:
                env.render(
                    save_path=config.LOGS_DIR / f"episode_{episode}.png"
                )
                render_counter += 1

        # Periodic logging
        if (episode + 1) % 100 == 0:
            recent_stats = episode_stats[-100:]
            avg_reward = np.mean([s["reward"] for s in recent_stats])
            avg_length = np.mean([s["length"] for s in recent_stats])
            success_rate = np.mean([s["success"] for s in recent_stats])

            print(f"\nEpisode {episode + 1}/{args.num_episodes}")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Avg length: {avg_length:.1f}")
            print(f"  Success rate: {success_rate:.1%}")

    # Save data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collector.save_data(output_path)

    # Print final statistics
    print("\n" + "="*50)
    print("Data Collection Complete!")
    print("="*50)

    data = collector.get_data()
    print(f"Total transitions: {len(data['observations'])}")
    print(f"Observation shape: {data['observations'].shape}")
    print(f"Action shape: {data['actions'].shape}")

    avg_reward = np.mean([s["reward"] for s in episode_stats])
    avg_length = np.mean([s["length"] for s in episode_stats])
    success_rate = np.mean([s["success"] for s in episode_stats])

    print(f"\nOverall Statistics:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average episode length: {avg_length:.1f}")
    print(f"  Success rate: {success_rate:.1%}")

    # Split and save train/val data
    print("\nSplitting data into train/val sets...")

    num_samples = len(data["observations"])
    num_train = int(num_samples * config.TRAINING_CONFIG["train_ratio"])

    # Shuffle indices
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Save train data
    train_data = {
        key: value[train_indices]
        for key, value in data.items()
    }
    np.savez_compressed(config.DATA_PATHS["train_data"], **train_data)
    print(f"Saved {len(train_indices)} training samples to {config.DATA_PATHS['train_data']}")

    # Save validation data
    val_data = {
        key: value[val_indices]
        for key, value in data.items()
    }
    np.savez_compressed(config.DATA_PATHS["val_data"], **val_data)
    print(f"Saved {len(val_indices)} validation samples to {config.DATA_PATHS['val_data']}")


if __name__ == "__main__":
    main()