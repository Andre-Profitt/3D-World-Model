"""
Simple 3D Navigation Environment with basic physics.

A drone/point-mass navigates in 3D space to reach a goal while avoiding boundaries.
No external physics engine required - pure numpy implementation.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Literal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from PIL import Image


class Simple3DNavEnv:
    """
    3D navigation environment with simple physics.

    State: [x, y, z, vx, vy, vz] - position and velocity
    Action: [ax, ay, az] - acceleration commands
    Reward: Based on distance to goal and collisions
    """

    def __init__(
        self,
        world_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        dt: float = 0.05,
        max_velocity: float = 2.0,
        max_acceleration: float = 5.0,
        goal_radius: float = 0.5,
        seed: Optional[int] = None,
        obs_type: Literal["state", "image", "both"] = "state",
        image_size: Tuple[int, int] = (64, 64),
        camera_mode: Literal["top_down", "agent_centric", "fixed_3d"] = "top_down",
        grayscale: bool = True,
    ):
        """
        Initialize the 3D navigation environment.

        Args:
            world_size: (width, depth, height) of the world box
            dt: Physics timestep
            max_velocity: Maximum velocity magnitude
            max_acceleration: Maximum acceleration magnitude
            goal_radius: Distance to goal for success
            seed: Random seed
            obs_type: Type of observation ("state", "image", "both")
            image_size: Size of image observations (height, width)
            camera_mode: Camera perspective mode
            grayscale: Whether to use grayscale images
        """
        self.world_size = np.array(world_size)
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.goal_radius = goal_radius

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Visual observation parameters
        self.obs_type = obs_type
        self.image_size = image_size
        self.camera_mode = camera_mode
        self.grayscale = grayscale
        self.image_channels = 1 if grayscale else 3

        # State dimensions
        self.state_dim = 6  # [x, y, z, vx, vy, vz]
        self.action_dim = 3  # [ax, ay, az]

        # Observation dimensions depend on type
        if obs_type == "state":
            self.obs_dim = 9    # state + goal position
        elif obs_type == "image":
            self.obs_dim = (self.image_channels, *image_size)
        else:  # both
            self.obs_dim = (9, (self.image_channels, *image_size))

        # Current state
        self.state = None
        self.goal = None
        self.steps = 0
        self.max_steps = 500

        # Tracking for visualization
        self.trajectory = []

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            observation: Initial observation
        """
        # Random initial position (avoid boundaries)
        margin = 1.0
        position = self.rng.uniform(
            low=margin,
            high=self.world_size - margin,
            size=3
        )

        # Start with small random velocity
        velocity = self.rng.uniform(-0.5, 0.5, size=3)

        self.state = np.concatenate([position, velocity])

        # Random goal position (avoid boundaries)
        self.goal = self.rng.uniform(
            low=margin,
            high=self.world_size - margin,
            size=3
        )

        # Ensure goal is not too close to start
        while np.linalg.norm(self.goal - position) < 2.0:
            self.goal = self.rng.uniform(
                low=margin,
                high=self.world_size - margin,
                size=3
            )

        self.steps = 0
        self.trajectory = [position.copy()]

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Acceleration command [ax, ay, az]

        Returns:
            observation: Next observation
            reward: Reward signal
            done: Episode termination flag
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping")

        # Clip action to maximum acceleration
        action = np.clip(action, -self.max_acceleration, self.max_acceleration)

        # Extract current state
        position = self.state[:3]
        velocity = self.state[3:6]

        # Update velocity (with drag for stability)
        drag = 0.02
        velocity = velocity * (1 - drag) + action * self.dt

        # Clip velocity to maximum
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude > self.max_velocity:
            velocity = velocity * (self.max_velocity / vel_magnitude)

        # Update position
        position = position + velocity * self.dt

        # Handle boundaries (elastic collision)
        for i in range(3):
            if position[i] <= 0:
                position[i] = 0
                velocity[i] = abs(velocity[i]) * 0.8  # Damped bounce
            elif position[i] >= self.world_size[i]:
                position[i] = self.world_size[i]
                velocity[i] = -abs(velocity[i]) * 0.8  # Damped bounce

        # Update state
        self.state = np.concatenate([position, velocity])
        self.trajectory.append(position.copy())

        # Compute reward
        dist_to_goal = np.linalg.norm(position - self.goal)

        # Dense reward: negative distance to goal
        reward = -dist_to_goal * 0.01

        # Bonus for reaching goal
        if dist_to_goal < self.goal_radius:
            reward += 10.0

        # Penalty for hitting boundaries
        if np.any(position <= 0.1) or np.any(position >= self.world_size - 0.1):
            reward -= 1.0

        # Small penalty for energy usage
        reward -= np.linalg.norm(action) * 0.001

        # Check termination
        self.steps += 1
        done = (
            dist_to_goal < self.goal_radius or  # Reached goal
            self.steps >= self.max_steps         # Timeout
        )

        # Additional info
        info = {
            "state": self.state.copy(),
            "goal": self.goal.copy(),
            "dist_to_goal": dist_to_goal,
            "success": dist_to_goal < self.goal_radius,
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """
        Get current observation based on observation type.

        Returns:
            observation: State vector, image, or both
        """
        if self.obs_type == "state":
            return np.concatenate([self.state, self.goal])
        elif self.obs_type == "image":
            return self._get_image_obs()
        else:  # both
            state_obs = np.concatenate([self.state, self.goal])
            image_obs = self._get_image_obs()
            return {"state": state_obs, "image": image_obs}

    def _get_image_obs(self) -> np.ndarray:
        """
        Render current state as an image observation.

        Returns:
            image: Image array of shape (C, H, W)
        """
        if self.camera_mode == "top_down":
            image = self._render_top_down()
        elif self.camera_mode == "agent_centric":
            image = self._render_agent_centric()
        elif self.camera_mode == "fixed_3d":
            image = self._render_3d_view()
        else:
            raise ValueError(f"Unknown camera mode: {self.camera_mode}")

        return image

    def _render_top_down(self) -> np.ndarray:
        """
        Render top-down 2D view of the environment.

        Returns:
            image: Rendered image array
        """
        fig, ax = plt.subplots(figsize=(4, 4), dpi=self.image_size[0]//4)

        # Set up the plot
        ax.set_xlim(0, self.world_size[0])
        ax.set_ylim(0, self.world_size[1])
        ax.set_aspect('equal')
        ax.axis('off')

        # Background
        ax.add_patch(patches.Rectangle((0, 0), self.world_size[0], self.world_size[1],
                                      facecolor='white', edgecolor='black', linewidth=2))

        # Draw agent (circle)
        position = self.state[:3]
        agent_circle = patches.Circle((position[0], position[1]), 0.3,
                                     facecolor='blue', edgecolor='darkblue', linewidth=2)
        ax.add_patch(agent_circle)

        # Draw velocity vector
        velocity = self.state[3:6]
        if np.linalg.norm(velocity[:2]) > 0.01:
            ax.arrow(position[0], position[1], velocity[0]*0.5, velocity[1]*0.5,
                    head_width=0.2, head_length=0.1, fc='cyan', ec='cyan', alpha=0.7)

        # Draw goal (star)
        ax.plot(self.goal[0], self.goal[1], 'g*', markersize=20)

        # Draw goal radius
        goal_circle = patches.Circle((self.goal[0], self.goal[1]), self.goal_radius,
                                    facecolor='none', edgecolor='green', linestyle='--', linewidth=1)
        ax.add_patch(goal_circle)

        # Draw trajectory
        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3, linewidth=1)

        # Height indicator (color or size based on z)
        z_normalized = position[2] / self.world_size[2]
        ax.text(position[0], position[1] - 0.5, f'z:{position[2]:.1f}',
               fontsize=8, ha='center', alpha=0.7)

        # Convert to image array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        # Resize to target size
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image[:, :, :3])  # Remove alpha
        pil_image = pil_image.resize(self.image_size, PILImage.LANCZOS)

        # Convert to numpy array
        image = np.array(pil_image)

        # Convert to grayscale if needed
        if self.grayscale:
            image = np.mean(image, axis=2, keepdims=False).astype(np.uint8)
            image = image[np.newaxis, :, :]  # Add channel dimension
        else:
            image = image.transpose(2, 0, 1)  # HWC to CHW

        return image.astype(np.float32) / 255.0

    def _render_agent_centric(self) -> np.ndarray:
        """
        Render agent-centric view (ego-centric).

        Returns:
            image: Rendered image array
        """
        fig, ax = plt.subplots(figsize=(4, 4), dpi=self.image_size[0]//4)

        # Agent is at the center
        view_range = 5.0
        position = self.state[:3]

        # Set up the plot centered on agent
        ax.set_xlim(position[0] - view_range, position[0] + view_range)
        ax.set_ylim(position[1] - view_range, position[1] + view_range)
        ax.set_aspect('equal')
        ax.axis('off')

        # Background
        ax.add_patch(patches.Rectangle((position[0] - view_range, position[1] - view_range),
                                      2 * view_range, 2 * view_range,
                                      facecolor='white', edgecolor='gray', linewidth=1))

        # Draw world boundaries if visible
        if position[0] - view_range < 0:
            ax.axvline(x=0, color='black', linewidth=2)
        if position[0] + view_range > self.world_size[0]:
            ax.axvline(x=self.world_size[0], color='black', linewidth=2)
        if position[1] - view_range < 0:
            ax.axhline(y=0, color='black', linewidth=2)
        if position[1] + view_range > self.world_size[1]:
            ax.axhline(y=self.world_size[1], color='black', linewidth=2)

        # Draw agent at center
        agent_circle = patches.Circle((position[0], position[1]), 0.3,
                                     facecolor='blue', edgecolor='darkblue', linewidth=2)
        ax.add_patch(agent_circle)

        # Draw velocity vector
        velocity = self.state[3:6]
        if np.linalg.norm(velocity[:2]) > 0.01:
            ax.arrow(position[0], position[1], velocity[0], velocity[1],
                    head_width=0.2, head_length=0.1, fc='cyan', ec='cyan', alpha=0.7)

        # Draw goal if visible
        goal_dist = np.linalg.norm(self.goal[:2] - position[:2])
        if goal_dist < view_range * 1.5:
            ax.plot(self.goal[0], self.goal[1], 'g*', markersize=20)
            goal_circle = patches.Circle((self.goal[0], self.goal[1]), self.goal_radius,
                                        facecolor='none', edgecolor='green', linestyle='--', linewidth=1)
            ax.add_patch(goal_circle)

        # Draw recent trajectory
        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory[-20:])  # Last 20 points
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3, linewidth=1)

        # Convert to image array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        # Resize and process
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image[:, :, :3])
        pil_image = pil_image.resize(self.image_size, PILImage.LANCZOS)
        image = np.array(pil_image)

        if self.grayscale:
            image = np.mean(image, axis=2, keepdims=False).astype(np.uint8)
            image = image[np.newaxis, :, :]
        else:
            image = image.transpose(2, 0, 1)

        return image.astype(np.float32) / 255.0

    def _render_3d_view(self) -> np.ndarray:
        """
        Render 3D perspective view.

        Returns:
            image: Rendered image array
        """
        fig = plt.figure(figsize=(4, 4), dpi=self.image_size[0]//4)
        ax = fig.add_subplot(111, projection='3d')

        # Set view angle
        ax.view_init(elev=20, azim=45)

        # Plot world boundaries
        self._plot_box(ax, [0, 0, 0], self.world_size, alpha=0.1)

        # Plot trajectory
        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   'b-', alpha=0.5, linewidth=2)

        # Plot agent
        position = self.state[:3]
        ax.scatter(position[0], position[1], position[2],
                  c='blue', s=100, marker='o')

        # Plot velocity
        velocity = self.state[3:6]
        ax.quiver(position[0], position[1], position[2],
                 velocity[0], velocity[1], velocity[2],
                 length=0.5, color='cyan', alpha=0.7)

        # Plot goal
        ax.scatter(self.goal[0], self.goal[1], self.goal[2],
                  c='green', s=200, marker='*')

        # Goal radius
        self._plot_sphere(ax, self.goal, self.goal_radius, color='green', alpha=0.2)

        # Remove labels for cleaner image
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)

        # Set limits
        ax.set_xlim([0, self.world_size[0]])
        ax.set_ylim([0, self.world_size[1]])
        ax.set_zlim([0, self.world_size[2]])

        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)

        # Resize and process
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image[:, :, :3])
        pil_image = pil_image.resize(self.image_size, PILImage.LANCZOS)
        image = np.array(pil_image)

        if self.grayscale:
            image = np.mean(image, axis=2, keepdims=False).astype(np.uint8)
            image = image[np.newaxis, :, :]
        else:
            image = image.transpose(2, 0, 1)

        return image.astype(np.float32) / 255.0

    def render(self, mode: str = "matplotlib", save_path: Optional[str] = None):
        """
        Render the current state of the environment.

        Args:
            mode: Rendering mode (currently only "matplotlib")
            save_path: Optional path to save the figure
        """
        if mode != "matplotlib":
            raise NotImplementedError(f"Rendering mode '{mode}' not supported")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot world boundaries
        self._plot_box(ax, [0, 0, 0], self.world_size, alpha=0.1)

        # Plot trajectory
        if len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                'b-',
                alpha=0.5,
                linewidth=2,
                label='Trajectory'
            )

        # Plot current position
        position = self.state[:3]
        ax.scatter(
            position[0], position[1], position[2],
            c='blue', s=100, marker='o', label='Agent'
        )

        # Plot velocity vector
        velocity = self.state[3:6]
        ax.quiver(
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2],
            length=0.5, color='cyan', alpha=0.7
        )

        # Plot goal
        ax.scatter(
            self.goal[0], self.goal[1], self.goal[2],
            c='green', s=200, marker='*', label='Goal'
        )

        # Goal radius sphere
        self._plot_sphere(ax, self.goal, self.goal_radius, color='green', alpha=0.2)

        # Labels and formatting
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Navigation Environment')
        ax.legend()

        # Set equal aspect ratio
        ax.set_xlim([0, self.world_size[0]])
        ax.set_ylim([0, self.world_size[1]])
        ax.set_zlim([0, self.world_size[2]])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def _plot_box(self, ax, origin, size, **kwargs):
        """Plot a 3D box."""
        ox, oy, oz = origin
        sx, sy, sz = size

        # Define the vertices of the box
        vertices = [
            [ox, oy, oz],
            [ox + sx, oy, oz],
            [ox + sx, oy + sy, oz],
            [ox, oy + sy, oz],
            [ox, oy, oz + sz],
            [ox + sx, oy, oz + sz],
            [ox + sx, oy + sy, oz + sz],
            [ox, oy + sy, oz + sz],
        ]

        # Define the 12 edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]

        for edge in edges:
            points = [vertices[edge[0]], vertices[edge[1]]]
            ax.plot3D(
                [points[0][0], points[1][0]],
                [points[0][1], points[1][1]],
                [points[0][2], points[1][2]],
                'k-', **kwargs
            )

    def _plot_sphere(self, ax, center, radius, **kwargs):
        """Plot a sphere."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, **kwargs)

    @property
    def observation_space_shape(self):
        """Get observation space shape."""
        return (self.obs_dim,)

    @property
    def action_space_shape(self):
        """Get action space shape."""
        return (self.action_dim,)


if __name__ == "__main__":
    # Test the environment
    env = Simple3DNavEnv(seed=42)

    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")

    # Run a few random steps
    for i in range(10):
        action = env.rng.uniform(-1, 1, size=3)
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}, dist={info['dist_to_goal']:.3f}")

        if done:
            break

    # Render the final state
    env.render()