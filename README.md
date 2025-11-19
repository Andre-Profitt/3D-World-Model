# 3D World Model

A complete implementation of a 3D world model with Model Predictive Control (MPC) for navigation tasks. The system learns environmental dynamics from experience and uses the learned model for planning optimal trajectories.

## Features

- **3D Physics Simulation**: Built on PyBullet for realistic rigid body dynamics.
- **World Model Learning**:
    - **Vector State**: MLP-based dynamics model.
    - **Visual Observations**: CNN-based encoder/decoder for learning from pixels.
    - **Latent Dynamics**: V-M-C architecture (Vision-Model-Controller) planning in latent space.
    - **Ensemble Models**: Uncertainty estimation using bootstrap ensembles.
    - **Stochastic Dynamics**: Probabilistic modeling (Gaussian) for aleatoric uncertainty.
- **Model-Based Planning**:
    - **MPC Controller**: Model Predictive Control with Random Shooting and CEM.
    - **Risk-Sensitive Planning**: Avoids uncertain regions using ensemble variance or stochastic variance.
- **Evaluation Suite**: Comprehensive tools for measuring planning horizon and reconstruction quality.

## State vs Latent Planning
The system supports planning in both raw state space and learned latent space:

| Feature | Vector State | Latent Space (V-M-C) |
|---------|--------------|---------------------|
| **Input** | 9D Vector (Pos, Vel, Goal) | 64x64 Image (or Vector) |
| **Representation** | Explicit State | Learned 16D Latent |
| **Dynamics** | MLP | Latent MLP |
| **Planning** | MPC on State | MPC on Latent |
| **Use Case** | Low-dim, fully observable | High-dim, visual, partial obs |

> [!NOTE]
> For 9D vector observations, the "latent" space is a re-representation (e.g., to 16D). For high-dimensional inputs like images, it provides true compression.

## Architecture

### Components

1. **Environment** (`envs/simple_3d_nav.py`)
   - 3D navigation task with position/velocity dynamics
   - Goal-reaching with collision avoidance
   - Configurable physics parameters

2. **World Model** (`models/world_model.py`, `models/latent_world_model.py`)
   - Predicts next observation and reward: `M(obs_t, a_t) → (obs_{t+1}, r_t)`
   - Supports single models, ensembles, and stochastic (Gaussian) models
   - Optional delta prediction for improved learning

3. **MPC Controller** (`models/mpc_controller.py`)
   - Random shooting or Cross-Entropy Method (CEM)
   - Evaluates action sequences using the world model (state or latent)
   - Receding horizon control
   - Risk-sensitive scoring: `mean - lambda * std`

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run an Experiment
The easiest way to get started is to run a full experiment pipeline:

```bash
# Run a standard vector-state experiment
python experiments/run_experiment.py --name demo_vector --mode vector

# Run a visual world model experiment
python experiments/run_experiment.py --name demo_visual --mode visual
```

### 3. Manual Workflow

**Data Collection**
```bash
python scripts/collect_data.py --num_episodes 50
```

**Training**
```bash
# Train standard world model
python training/train_world_model.py

# Train with multi-step loss
python training/train_world_model.py --use_multistep

# Train visual world model
python training/train_visual_world_model.py
```

**Evaluation**
```bash
# Evaluate MPC agent
python scripts/run_mpc_agent.py --render

# Evaluate visual agent
python scripts/run_mpc_agent.py --use_visual --render
```

## Latent Space MPC (V-M-C)
The system supports planning in a learned latent space, which is essential for high-dimensional observations like images.

### 1. Collect Data
```bash
python scripts/collect_data.py --num_episodes 1000 --output data/3d_nav_buffer.npz
```

### 2. Train Autoencoder
```bash
python training/train_autoencoder.py \
    --data_path data/3d_nav_buffer.npz \
    --latent_dim 6
```

### 3. Train Latent World Model
```bash
python training/train_latent_world_model.py \
    --data_path data/3d_nav_buffer.npz \
    --encoder_path weights/encoder.pt \
    --decoder_path weights/decoder.pt \
    --latent_dim 6
```

### 4. Run MPC in Latent Space
```bash
python scripts/run_mpc_agent.py \
    --use_latent \
    --encoder_path weights/encoder.pt \
    --latent_model_path weights/latent_world_model.pt \
    --use_cem \
    --compare_baselines
```

## Stochastic World Model

For environments with uncertainty, you can train a stochastic world model that predicts distributions over future states. This is available for both **State Space** and **Latent Space**.

### 1. State-Space Stochastic Model

Train a probabilistic model on raw vector observations:

```bash
# Train stochastic model
python training/train_stochastic_world_model.py \
    --num_epochs 100 \
    --batch_size 256

# Train ensemble of stochastic models
python training/train_stochastic_world_model.py \
    --use_ensemble \
    --ensemble_size 5
```

### 2. Latent-Space Stochastic Model

Train a probabilistic model in the learned latent space:

```bash
# Train stochastic latent model
python training/train_latent_world_model.py \
    --stochastic \
    --encoder_path weights/encoder.pt \
    --decoder_path weights/decoder.pt \
    --latent_dim 6
```

### 3. Planning with Uncertainty

Run MPC with stochastic rollouts. The planner samples multiple futures for each action sequence to estimate risk.

```bash
# Run MPC with stochastic state-space model
python scripts/run_mpc_agent.py \
    --use_stochastic \
    --stochastic_rollouts 20 \
    --lambda_risk 1.0

# Run MPC with stochastic latent model
python scripts/run_mpc_agent.py \
    --use_latent \
    --use_stochastic \
    --latent_model_path weights/latent_world_model_stochastic.pt \
    --stochastic_rollouts 20 \
    --lambda_risk 1.0
```

**Key Arguments:**
- `--use_stochastic`: Enable stochastic model usage.
- `--stochastic_rollouts N`: Sample N distinct futures for each candidate action sequence (Monte Carlo sampling).
- `--lambda_risk`: Penalty weight for variance in returns (`score = mean - lambda * std`). Higher values make the agent more risk-averse.

## Performance Comparison (Approximate)

| Model Type | Planning Space | Success Rate | Mean Reward | Notes |
|------------|----------------|--------------|-------------|-------|
| **State-Space WM** | Vector Obs (9D) | ~90% | -20 | Fastest, baseline |
| **Latent WM** | Latent (6D) | ~88% | -21 | Comparable performance |
| **Visual WM** | Latent Image (64D) | ~75% | -30 | Harder problem, heavier compute |

## Configuration
All hyperparameters are defined in `wm_config.py`. You can adjust model architectures, training settings, and environment parameters there.

## Project Structure
- `envs/`: 3D navigation environment.
- `models/`: Neural network architectures (WorldModel, Encoder, Decoder, etc.).
- `training/`: Training scripts.
- `scripts/`: Data collection and evaluation scripts.
- `experiments/`: Experiment runners and configurations.
- `data/`: Collected trajectories.
- `weights/`: Saved model checkpoints.
- `logs/`: Tensorboard logs and experiment results.

## Citation
If you use this code, please cite:
```
@misc{3dworldmodel2024,
  author = {Andre Profitt},
  title = {3D World Model with MPC},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Andre-Profitt/3D-World-Model}},
}
```

## Reproducibility

To ensure reproducible results, all training and evaluation scripts support a `--seed` argument. This sets the random seed for Python, NumPy, and PyTorch.

```bash
# Example: Train with a specific seed
python training/train_world_model.py --seed 42
```

## License

MIT License - see LICENSE file for details.