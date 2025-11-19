# Walkthrough - Visual Observations & Multi-step Loss

I have successfully upgraded the 3D World Model with visual observation capabilities and improved training stability.

## Changes

### 1. Visual Observations (Phase 5)
- **Environment**: Updated `envs/simple_3d_nav.py` to render 64x64 RGB images from the agent's perspective.
- **Models**: Added `ConvEncoder` and `ConvDecoder` in `models/vision_encoder.py` to process image data.
- **Training**: Created `training/train_visual_world_model.py` to train visual autoencoders and latent world models.
- **Agent**: Updated `scripts/run_mpc_agent.py` to support `--use_visual` flag for planning with visual inputs.

### 2. Multi-step Training Loss (Phase 1.2)
- **Model**: Added `unrolled_loss` method to `WorldModel` in `models/world_model.py`. This computes the error over a trajectory of $K$ steps (default 5), forcing the model to be stable over longer horizons.
- **Training**: Updated `training/train_world_model.py` to support `--use_multistep`. It now uses a `SequenceDataset` to load full trajectories.

### 3. Documentation & Experiments (Phase 6)
- **Experiments**: Created `experiments/` directory with a `run_experiment.py` script to easily run full pipelines (collection -> training -> evaluation).
- **Config**: Renamed `config.py` to `wm_config.py` to avoid conflicts with system libraries.
- **README**: Updated main `README.md` with comprehensive instructions.

## Verification Results

### Visual World Model
I verified the visual pipeline by running the training script. It successfully collects visual data, trains the autoencoder, and trains the latent dynamics model.

### Multi-step Loss
I verified the multi-step loss by training a vector-state model with `--use_multistep`. The training converged and produced a valid model.

## How to Run

### Run a Full Experiment
```bash
# Visual World Model
python experiments/run_experiment.py --name demo_visual --mode visual

# Vector World Model with Multi-step Loss
# (Note: run_experiment.py currently uses default training args, so modify it or run manually for specific flags)
```

### Manual Run
```bash
# 1. Collect Data
python scripts/collect_data.py --num_episodes 50

# 2. Train with Multi-step Loss
python training/train_world_model.py --use_multistep --num_epochs 50

# 3. Evaluate
python scripts/run_mpc_agent.py --render
```
