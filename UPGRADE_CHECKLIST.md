# World Model Upgrade Checklist

Transform the current 3D MPC controller into a true modern world model (V-M-C architecture with latent dynamics).

## Target Definition
A system that takes sequences of partial, high-dimensional observations and actions, learns a latent representation of the 3D world, models how that latent evolves over time (including reward), and uses that model to plan/control with long-horizon fidelity.

---

## Phase 1: Solidify Current Vector-State World Model

### 1.1 Long-Horizon Evaluation
- [ ] Create `scripts/eval_world_model_rollouts.py`
  - [ ] Sample 100+ test episodes (new seeds, never seen in training)
  - [ ] For each episode, roll out both real env and world model for 50-100 steps
  - [ ] Compute L2 error on observations over time
  - [ ] Compute reward error over time
  - [ ] Track success discrepancy (goal reached in model vs reality)
  - [ ] Generate plots: error vs timestep, error vs rollout length
  - [ ] Save metrics as JSON and plots as PNG

### 1.2 Improve Dynamics Model Robustness
- [ ] Enable delta prediction by default in `models/world_model.py`
  - [ ] Modify forward() to predict Δobs = obs_{t+1} - obs_t
  - [ ] Add reconstruction: obs_{t+1} = obs_t + Δobs_pred
  - [ ] Update loss function to compute on deltas

- [ ] Add multi-step training loss
  - [ ] Implement `unrolled_loss()` method in WorldModel
  - [ ] Unroll K steps (K=5 default) using true actions
  - [ ] Compare predicted obs_{t+1..t+K} against real
  - [ ] Weight multi-step loss with single-step loss

- [ ] Fully wire ensemble support
  - [ ] Ensure ensemble trains N models with different seeds
  - [ ] Add bootstrap sampling option for training data
  - [ ] Expose mean and variance predictions
  - [ ] Update MPC to use uncertainty (penalize high variance)

---

## Phase 2: Add Explicit Latent Representation (V)

### 2.1 Encoder/Decoder Modules
- [ ] Create `models/encoder_decoder.py`
  ```python
  class Encoder(nn.Module):
      def __init__(self, obs_dim=9, latent_dim=16, hidden_dim=128):
          # MLP: obs → z

  class Decoder(nn.Module):
      def __init__(self, latent_dim=16, obs_dim=9, hidden_dim=128):
          # MLP: z → reconstructed obs
  ```

### 2.2 Autoencoder Training
- [ ] Create `training/train_autoencoder.py`
  - [ ] Load observations from experience buffer
  - [ ] Implement reconstruction loss: MSE(decoder(encoder(obs)), obs)
  - [ ] Add train/val split
  - [ ] Implement early stopping based on val loss
  - [ ] Save encoder/decoder weights separately

- [ ] Update `config.py` with latent settings
  ```python
  LATENT_CONFIG = {
      "latent_dim": 16,
      "hidden_dim": 128,
      "ae_learning_rate": 1e-3,
      "ae_batch_size": 256,
      "ae_epochs": 50,
  }
  ```

---

## Phase 3: Move World Model to Latent Space

### 3.1 Latent Dataset Builder
- [ ] Create `scripts/build_latent_dataset.py`
  - [ ] Load experience buffer (obs_t, actions, obs_{t+1}, rewards)
  - [ ] Load trained encoder
  - [ ] Compute z_t = encoder(obs_t), z_{t+1} = encoder(obs_{t+1})
  - [ ] Save (z_t, actions, z_{t+1}, rewards) as new dataset
  - [ ] Add option to normalize latents

### 3.2 Latent World Model
- [ ] Create `models/latent_world_model.py`
  ```python
  class LatentWorldModel(nn.Module):
      def __init__(self, latent_dim=16, action_dim=3):
          # Input: [z_t, action_t]
          # Output: [z_{t+1}, r_t] or [Δz, r_t]
  ```

- [ ] Create `training/train_latent_world_model.py`
  - [ ] Load latent dataset
  - [ ] Train with MSE on latent transitions
  - [ ] Log multi-step rollout errors
  - [ ] Decode periodically to check reconstruction quality

### 3.3 Update MPC for Latent Space
- [ ] Modify `models/mpc_controller.py`
  - [ ] Accept encoder in __init__
  - [ ] Replace obs_t with z_t = encoder(obs_t) at planning start
  - [ ] Roll out latent world model: z_next, r = model(z, a)
  - [ ] Optional: decode for cost shaping

- [ ] Update `scripts/run_mpc_agent.py`
  - [ ] Load encoder along with world model
  - [ ] Pass encoder to MPC controller
  - [ ] Add latent space visualization option

---

## Phase 4: Stochastic Dynamics & Uncertainty

### 4.1 Stochastic Latent World Model
- [ ] Extend LatentWorldModel to output distributions
  - [ ] Output: [μ_z, logσ²_z, r] (size: 2*latent_dim + 1)
  - [ ] Implement reparameterization trick
  - [ ] Add KL divergence regularization

- [ ] Update training for stochastic models
  - [ ] Implement ELBO loss (reconstruction + KL)
  - [ ] Add variance scheduling (start deterministic, increase stochasticity)
  - [ ] Log uncertainty metrics

### 4.2 Risk-Aware Planning
- [ ] Update MPC controller for uncertainty
  - [ ] Sample K trajectories per action sequence
  - [ ] Compute mean and variance of returns
  - [ ] Implement risk-sensitive objectives:
    - [ ] Mean - λ*std (risk-averse)
    - [ ] CVaR (conditional value at risk)
    - [ ] Upper confidence bound
  - [ ] Add config for risk preference

---

## Phase 5: Visual 3D World Modeling

### 5.1 Camera/Depth Observations
- [ ] Extend `envs/simple_3d_nav.py`
  - [ ] Add `get_camera_obs()` method
  - [ ] Render 64x64 RGB or depth from agent POV
  - [ ] Add overhead camera option
  - [ ] Support multiple camera angles

- [ ] Update config for observation types
  ```python
  OBS_CONFIG = {
      "type": "image",  # "state", "image", "state+image"
      "image_size": (64, 64),
      "channels": 3,  # RGB
  }
  ```

### 5.2 Convolutional Encoder/Decoder
- [ ] Create `models/visual_encoder.py`
  ```python
  class ConvEncoder(nn.Module):
      # CNN: image → z_visual (dim 32-64)

  class ConvDecoder(nn.Module):
      # Deconv: z_visual → reconstructed image
  ```

- [ ] Training options
  - [ ] Train visual autoencoder separately
  - [ ] Joint training with state encoder
  - [ ] Hybrid latent: z = concat(z_visual, z_state)

### 5.3 Visual World Model
- [ ] Implement visual-only option
  - [ ] World model on z_visual only
  - [ ] Condition on goal via FiLM or concatenation

- [ ] Implement hybrid option
  - [ ] Combine visual and state latents
  - [ ] Learn joint dynamics

- [ ] Update evaluation
  - [ ] Compare predicted vs true frames
  - [ ] Compute PSNR, SSIM metrics
  - [ ] Generate video predictions

---

## Phase 6: Documentation & Experiments

### 6.1 Experiment Suite
- [ ] Create `experiments/` folder
- [ ] `vector_state_world_model.md`
  - [ ] Performance metrics
  - [ ] Error vs horizon plots
  - [ ] MPC vs baseline comparisons

- [ ] `latent_world_model.md`
  - [ ] Benefits of latent dynamics
  - [ ] Compression ratio analysis
  - [ ] Reconstruction quality

- [ ] `visual_world_model.md`
  - [ ] Frame predictions
  - [ ] Control from pixels
  - [ ] Ablation studies

### 6.2 Documentation Updates
- [ ] Update README with V-M-C diagram
- [ ] Add literature connections
  - [ ] Reference Ha & Schmidhuber 2018
  - [ ] Reference Dreamer papers
  - [ ] Explain latent dynamics benefits

- [ ] Create tutorial notebooks
  - [ ] `notebooks/01_vector_world_model.ipynb`
  - [ ] `notebooks/02_latent_dynamics.ipynb`
  - [ ] `notebooks/03_visual_control.ipynb`

---

## Implementation Order

### Quick Wins (1-2 days each)
1. Delta prediction (1.2)
2. Long-horizon evaluation (1.1)
3. Basic encoder/decoder (2.1)

### Core Upgrades (2-3 days each)
4. Autoencoder training (2.2)
5. Latent world model (3.2)
6. MPC in latent space (3.3)

### Advanced Features (3-5 days each)
7. Multi-step loss (1.2)
8. Stochastic models (4.1)
9. Visual observations (5.1-5.3)

### Polish (1-2 days)
10. Documentation & experiments (6.1-6.2)

---

## Success Metrics

### Phase 1 Complete When:
- [ ] World model maintains <10% error for 20+ steps
- [ ] Ensemble reduces prediction variance by 30%
- [ ] Multi-step training improves long-horizon by 2x

### Phase 2-3 Complete When:
- [ ] Autoencoder reconstruction error <0.01
- [ ] Latent world model matches/beats raw observation model
- [ ] MPC works entirely in latent space

### Phase 4 Complete When:
- [ ] Uncertainty estimates correlate with actual errors
- [ ] Risk-aware planning improves success rate by 10%
- [ ] Model signals when predictions are unreliable

### Phase 5 Complete When:
- [ ] Control from 64x64 images only
- [ ] Visual predictions accurate for 10+ frames
- [ ] Performance within 20% of state-based control

### Phase 6 Complete When:
- [ ] All experiments documented with plots
- [ ] Clear V-M-C architecture diagram
- [ ] Tutorial notebooks run end-to-end

---

## Notes

- Each checkbox should be a PR or commit
- Run evaluation after each major phase
- Keep backwards compatibility (flags for old behavior)
- Benchmark performance impact of each change