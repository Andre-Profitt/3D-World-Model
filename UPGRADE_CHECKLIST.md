# World Model Upgrade Checklist

Transform the current 3D MPC controller into a true modern world model (V-M-C architecture with latent dynamics).

## Target Definition
A system that takes sequences of partial, high-dimensional observations and actions, learns a latent representation of the 3D world, models how that latent evolves over time (including reward), and uses that model to plan/control with long-horizon fidelity.

---

## Phase 1: Solidify Current Vector-State World Model

### 1.1 Long-Horizon Evaluation
- [x] Create `scripts/eval_world_model_rollouts.py`
  - [x] Sample 100+ test episodes (new seeds, never seen in training)
  - [x] For each episode, roll out both real env and world model for 50-100 steps
  - [x] Compute L2 error on observations over time
  - [x] Compute reward error over time
  - [x] Track success discrepancy (goal reached in model vs reality)
  - [x] Generate plots: error vs timestep, error vs rollout length
  - [x] Save metrics as JSON and plots as PNG

### 1.2 Improve Dynamics Model Robustness
- [x] Implement Delta Prediction (predict $s_{t+1} - s_t$ instead of $s_{t+1}$)
  - Already in `WorldModel` class (`predict_delta=True`).
- [x] Multi-step Training Loss (Unrolled)
  - [x] Add `unrolled_loss` method to `WorldModel`.
  - [x] Update training loop to unroll model for $K$ steps.
  - [x] Minimize error over trajectory: $\mathcal{L} = \sum_{k=1}^K ||\hat{s}_{t+k} - s_{t+k}||^2$.hod in WorldModel
  - [ ] Unroll K steps (K=5 default) using true actions
  - [ ] Compare predicted obs_{t+1..t+K} against real
  - [ ] Weight multi-step loss with single-step loss

- [x] Fully wire ensemble support
  - [x] Ensure ensemble trains N models with different seeds
  - [x] Add bootstrap sampling option for training data
  - [x] Expose mean and variance predictions
  - [x] Update MPC to use uncertainty (penalize high variance)

---

## Phase 2: Add Explicit Latent Representation (V)

### 2.1 Encoder/Decoder Modules
- [x] Create `models/encoder_decoder.py`
  ```python
  class Encoder(nn.Module):
      def __init__(self, obs_dim=9, latent_dim=16, hidden_dim=128):
          # MLP: obs → z

  class Decoder(nn.Module):
      def __init__(self, latent_dim=16, obs_dim=9, hidden_dim=128):
          # MLP: z → reconstructed obs
  ```

### 2.2 Autoencoder Training
- [x] Create `training/train_autoencoder.py`
  - [x] Load observations from experience buffer
  - [x] Implement reconstruction loss: MSE(decoder(encoder(obs)), obs)
  - [x] Add train/val split
  - [x] Implement early stopping based on val loss
  - [x] Save encoder/decoder weights separately

- [x] Update `config.py` with latent settings
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
- [x] Create `scripts/build_latent_dataset.py` (Handled on-the-fly in training script)
  - [x] Load experience buffer (obs_t, actions, obs_{t+1}, rewards)
  - [x] Load trained encoder
  - [x] Compute z_t = encoder(obs_t), z_{t+1} = encoder(obs_{t+1})
  - [x] Save (z_t, actions, z_{t+1}, rewards) as new dataset
  - [x] Add option to normalize latents

### 3.2 Latent World Model
- [x] Create `models/latent_world_model.py`
  ```python
  class LatentWorldModel(nn.Module):
      def __init__(self, latent_dim=16, action_dim=3):
          # Input: [z_t, action_t]
          # Output: [z_{t+1}, r_t] or [Δz, r_t]
  ```

- [x] Create `training/train_latent_world_model.py`
  - [x] Load latent dataset
  - [x] Train with MSE on latent transitions
  - [x] Log multi-step rollout errors
  - [x] Decode periodically to check reconstruction quality

### 3.3 Update MPC for Latent Space
- [x] Modify `models/mpc_controller.py` (Via `LatentMPCWrapper`)
  - [x] Accept encoder in __init__
  - [x] Replace obs_t with z_t = encoder(obs_t) at planning start
  - [x] Roll out latent world model: z_next, r = model(z, a)
  - [x] Optional: decode for cost shaping

- [x] Update `scripts/run_mpc_agent.py`
  - [x] Load encoder along with world model
  - [x] Pass encoder to MPC controller
  - [x] Add latent space visualization option

---

## Phase 4: Stochastic Dynamics & Uncertainty

### 4.1 Stochastic Latent World Model
- [x] Extend LatentWorldModel to output distributions
  - [x] Output: [μ_z, logσ²_z, r] (size: 2*latent_dim + 1)
  - [x] Implement reparameterization trick
  - [x] Add KL divergence regularization

- [x] Update training for stochastic models
  - [x] Implement ELBO loss (reconstruction + KL)
  - [x] Add variance scheduling (start deterministic, increase stochasticity)
  - [x] Log uncertainty metrics

### 4.2 Risk-Aware Planning
- [x] Update MPC controller for uncertainty
  - [x] Sample K trajectories per action sequence
  - [x] Compute mean and variance of returns
  - [x] Implement risk-sensitive objectives:
    - [x] Mean - λ*std (risk-averse)
    - [ ] CVaR (conditional value at risk)
    - [ ] Upper confidence bound
  - [x] Add config for risk preference

---

## Phase 5: Visual 3D World Modeling

### 5.1 Camera/Depth Observations
- [x] Extend `envs/simple_3d_nav.py`
  - [x] Add `get_camera_obs()` method
  - [x] Render 64x64 RGB or depth from agent POV
  - [x] Add overhead camera option
  - [x] Support multiple camera angles

- [x] Update config for observation types
  ```python
  OBS_CONFIG = {
      "type": "image",  # "state", "image", "state+image"
      "image_size": (64, 64),
      "channels": 3,  # RGB
  }
  ```

### 5.2 Convolutional Encoder/Decoder
- [x] Create `models/visual_encoder.py`
  ```python
  class ConvEncoder(nn.Module):
      # CNN: image → z_visual (dim 32-64)

  class ConvDecoder(nn.Module):
      # Deconv: z_visual → reconstructed image
  ```

- [x] Training options
  - [x] Train visual autoencoder separately
  - [ ] Joint training with state encoder
  - [ ] Hybrid latent: z = concat(z_visual, z_state)

### 5.3 Visual World Model
- [x] Implement visual-only option
  - [x] World model on z_visual only
  - [ ] Condition on goal via FiLM or concatenation

- [ ] Implement hybrid option
  - [ ] Combine visual and state latents
  - [ ] Learn joint dynamics

- [x] Update evaluation
  - [ ] Compare predicted vs true frames
  - [ ] Compute PSNR, SSIM metrics
  - [ ] Generate video predictions
  - [x] Compare predicted vs true frames
  - [x] Compute PSNR, SSIM metrics
  - [x] Generate video predictions

---

## Phase 6: Documentation & Experiments

### 6.1 Experiment Suite
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
1. Delta prediction (1.2) [Done]
2. Long-horizon evaluation (1.1) [Done]
3. Basic encoder/decoder (2.1) [Done]

### Core Upgrades (2-3 days each)
4. Autoencoder training (2.2) [Done]
5. Latent world model (3.2) [Done]
6. MPC in latent space (3.3) [Done]

### Advanced Features (3-5 days each)
7. Multi-step loss (1.2) [Pending]
8. Stochastic models (4.1) [Done]
9. Visual observations (5.1-5.3) [Next]

### Polish (1-2 days)
10. Documentation & experiments (6.1-6.2) [Pending]

---

## Success Metrics

### Phase 1 Complete When:
- [x] World model maintains <10% error for 20+ steps
- [x] Ensemble reduces prediction variance by 30%
- [ ] Multi-step training improves long-horizon by 2x

### Phase 2-3 Complete When:
- [x] Autoencoder reconstruction error <0.01
- [x] Latent world model matches/beats raw observation model
- [x] MPC works entirely in latent space

### Phase 4 Complete When:
- [x] Uncertainty estimates correlate with actual errors
- [x] Risk-aware planning improves success rate by 10%
- [x] Model signals when predictions are unreliable

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