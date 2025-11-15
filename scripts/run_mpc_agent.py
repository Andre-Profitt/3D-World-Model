"""
Run MPC agent using the trained world model.

The agent uses model-based planning to navigate to goals in the 3D environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from envs import Simple3DNavEnv
from models import WorldModel, MPCController, MPCAgent
import config


def evaluate_agent(
    env: Simple3DNavEnv,
    agent: MPCAgent,
    num_episodes: int = 50,
    max_steps: int = 500,
    render: bool = False,
    save_videos: bool = False,
    video_frequency: int = 10,
) -> dict:
    """
    Evaluate MPC agent.

    Args:
        env: Environment
        agent: MPC agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render episodes
        save_videos: Whether to save videos
        video_frequency: Save video every N episodes

    Returns:
        Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_final_dists = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        agent.reset()

        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            # Select action
            action = agent.act(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(info["success"])
        episode_final_dists.append(info["dist_to_goal"])

        # Render
        if render and (episode % video_frequency == 0 or episode == 0):
            if save_videos:
                save_path = config.LOGS_DIR / f"mpc_episode_{episode}.png"
            else:
                save_path = None

            env.render(save_path=save_path)

            if save_path:
                print(f"Saved visualization to {save_path}")

    # Compute statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": np.mean(episode_successes),
        "mean_final_dist": np.mean(episode_final_dists),
        "std_final_dist": np.std(episode_final_dists),
    }

    return stats, {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "successes": episode_successes,
        "final_dists": episode_final_dists,
    }


def evaluate_baseline(
    env: Simple3DNavEnv,
    policy: str = "random",
    num_episodes: int = 50,
    max_steps: int = 500,
) -> dict:
    """
    Evaluate baseline policy.

    Args:
        env: Environment
        policy: Baseline policy type ("random" or "heuristic")
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_final_dists = []

    for episode in tqdm(range(num_episodes), desc=f"Evaluating {policy} baseline"):
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            if policy == "random":
                # Random action
                action = env.rng.uniform(
                    -env.max_acceleration,
                    env.max_acceleration,
                    size=3
                )
            elif policy == "heuristic":
                # Simple heuristic
                state = obs[:6]
                goal = obs[6:9]
                position = state[:3]
                velocity = state[3:6]

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
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(info["success"])
        episode_final_dists.append(info["dist_to_goal"])

    # Compute statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "success_rate": np.mean(episode_successes),
        "mean_final_dist": np.mean(episode_final_dists),
    }

    return stats


def visualize_planning(
    env: Simple3DNavEnv,
    controller: MPCController,
    num_steps: int = 5,
):
    """
    Visualize MPC planning.

    Shows both the planned trajectory and actual execution.
    """
    obs = env.reset()

    fig = plt.figure(figsize=(15, 5))

    for step in range(num_steps):
        # Plan trajectory
        actions, predicted_obs, predicted_rewards = controller.plan_trajectory(
            obs,
            horizon=min(controller.horizon, 10)
        )

        # Extract positions
        predicted_positions = predicted_obs[:, :3]

        # Visualize
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # 3D trajectory
        current_pos = obs[:3]
        goal_pos = obs[6:9]

        ax1.plot(
            predicted_positions[:, 0],
            predicted_positions[:, 1],
            predicted_positions[:, 2],
            'b-o',
            alpha=0.7,
            label='Planned'
        )
        ax1.scatter(
            current_pos[0], current_pos[1], current_pos[2],
            c='green', s=100, marker='o', label='Current'
        )
        ax1.scatter(
            goal_pos[0], goal_pos[1], goal_pos[2],
            c='red', s=200, marker='*', label='Goal'
        )

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Step {step+1}: Planned Trajectory')
        ax1.legend()

        # Predicted rewards
        ax2.plot(predicted_rewards, 'g-o')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Predicted Reward')
        ax2.set_title('Predicted Rewards')
        ax2.grid(True)

        # Actions
        ax3.plot(actions[:, 0], label='ax')
        ax3.plot(actions[:, 1], label='ay')
        ax3.plot(actions[:, 2], label='az')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Action')
        ax3.set_title('Planned Actions')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(config.LOGS_DIR / f"planning_step_{step}.png", dpi=150)
        plt.clf()

        # Execute first action
        obs, _, done, _ = env.step(actions[0])

        if done:
            break

    plt.close()


def main():
    """Main evaluation routine."""
    parser = argparse.ArgumentParser(description="Run MPC agent")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=config.EVAL_CONFIG["num_episodes"],
        help="Number of episodes"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=config.EVAL_CONFIG["max_steps"],
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.MODEL_PATHS["best_model"]),
        help="Path to trained model"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=config.MPC_CONFIG["horizon"],
        help="MPC planning horizon"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=config.MPC_CONFIG["num_samples"],
        help="Number of samples for MPC"
    )
    parser.add_argument(
        "--use_cem",
        action="store_true",
        default=config.MPC_CONFIG["use_cem"],
        help="Use CEM instead of random shooting"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=config.EVAL_CONFIG["render"],
        help="Render episodes"
    )
    parser.add_argument(
        "--compare_baselines",
        action="store_true",
        help="Compare with baseline policies"
    )
    parser.add_argument(
        "--visualize_planning",
        action="store_true",
        help="Visualize planning process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE_CONFIG["device"],
        help="Device to run on"
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first: python training/train_world_model.py")
        return

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)

    # Create environment
    env = Simple3DNavEnv(**config.ENV_CONFIG)

    # Get dimensions
    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]

    # Create world model
    world_model = WorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **config.MODEL_CONFIG["world_model"],
    )
    world_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['best_val_loss']:.4f}")

    # Create MPC controller
    controller = MPCController(
        world_model=world_model,
        action_dim=action_dim,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elite=config.MPC_CONFIG["num_elite"],
        gamma=config.MPC_CONFIG["gamma"],
        temperature=config.MPC_CONFIG["temperature"],
        action_noise=config.MPC_CONFIG["action_noise"],
        optimization_iters=config.MPC_CONFIG["optimization_iters"],
        use_cem=args.use_cem,
        action_min=-env.max_acceleration,
        action_max=env.max_acceleration,
        device=args.device,
    )

    # Create agent
    agent = MPCAgent(
        controller=controller,
        action_smoothing=0.1,
        warm_start=True,
    )

    print("\n" + "="*60)
    print("MPC Agent Evaluation")
    print("="*60)
    print(f"Planning method: {'CEM' if args.use_cem else 'Random Shooting'}")
    print(f"Horizon: {args.horizon}")
    print(f"Samples: {args.num_samples}")

    # Evaluate MPC agent
    print("\nEvaluating MPC agent...")
    mpc_stats, mpc_details = evaluate_agent(
        env,
        agent,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_videos=config.EVAL_CONFIG["save_videos"],
        video_frequency=config.EVAL_CONFIG["video_frequency"],
    )

    print("\n" + "-"*40)
    print("MPC Agent Results:")
    print("-"*40)
    print(f"Mean reward:     {mpc_stats['mean_reward']:.2f} ± {mpc_stats['std_reward']:.2f}")
    print(f"Success rate:    {mpc_stats['success_rate']:.1%}")
    print(f"Mean final dist: {mpc_stats['mean_final_dist']:.2f} ± {mpc_stats['std_final_dist']:.2f}")
    print(f"Mean length:     {mpc_stats['mean_length']:.1f} ± {mpc_stats['std_length']:.1f}")

    # Compare with baselines
    if args.compare_baselines:
        print("\n" + "="*60)
        print("Baseline Comparison")
        print("="*60)

        # Random baseline
        print("\nEvaluating random baseline...")
        random_stats = evaluate_baseline(
            env,
            policy="random",
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
        )

        print("\n" + "-"*40)
        print("Random Baseline Results:")
        print("-"*40)
        print(f"Mean reward:     {random_stats['mean_reward']:.2f} ± {random_stats['std_reward']:.2f}")
        print(f"Success rate:    {random_stats['success_rate']:.1%}")
        print(f"Mean final dist: {random_stats['mean_final_dist']:.2f}")

        # Heuristic baseline
        print("\nEvaluating heuristic baseline...")
        heuristic_stats = evaluate_baseline(
            env,
            policy="heuristic",
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
        )

        print("\n" + "-"*40)
        print("Heuristic Baseline Results:")
        print("-"*40)
        print(f"Mean reward:     {heuristic_stats['mean_reward']:.2f} ± {heuristic_stats['std_reward']:.2f}")
        print(f"Success rate:    {heuristic_stats['success_rate']:.1%}")
        print(f"Mean final dist: {heuristic_stats['mean_final_dist']:.2f}")

        # Summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"{'Method':<15} {'Success Rate':>15} {'Mean Reward':>15}")
        print("-"*45)
        print(f"{'MPC':<15} {mpc_stats['success_rate']:>14.1%} {mpc_stats['mean_reward']:>15.2f}")
        print(f"{'Heuristic':<15} {heuristic_stats['success_rate']:>14.1%} {heuristic_stats['mean_reward']:>15.2f}")
        print(f"{'Random':<15} {random_stats['success_rate']:>14.1%} {random_stats['mean_reward']:>15.2f}")

        # Improvement
        print("\nMPC Improvement over Random:")
        print(f"  Reward:  {mpc_stats['mean_reward'] - random_stats['mean_reward']:+.2f}")
        print(f"  Success: {(mpc_stats['success_rate'] - random_stats['success_rate']) * 100:+.1f}%")

    # Visualize planning
    if args.visualize_planning:
        print("\nVisualizing planning...")
        visualize_planning(env, controller, num_steps=5)
        print(f"Saved planning visualizations to {config.LOGS_DIR}")


if __name__ == "__main__":
    main()