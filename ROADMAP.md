# 3D World Model Development Roadmap

Engineering roadmap to achieve a modern, production-grade world model implementation with uncertainty-aware dynamics, latent representations, and visual observations.

## Current State (v0.1) ✅
- Basic 3D environment with physics
- Single deterministic world model
- MPC with random shooting/CEM
- Training pipeline

## Target State
A modern world-model system featuring:
- **Uncertainty-aware dynamics** (ensembles + stochastic models)
- **Quantitative model fidelity metrics** over long horizons
- **Explicit V-M-C architecture** (encoder → latent dynamics → controller)
- **Visual observation support** (learning from pixels)

---

## Milestone v0.2: Uncertainty & Evaluation
**Goal:** Make the system uncertainty-aware and rigorously evaluated
**Timeline:** 3-4 days
**Impact:** Can claim "production-grade world model with uncertainty"

### Task 1: Wire Ensemble World Model End-to-End

**Files to modify:**
- `config.py`
- `training/train_world_model.py`
- `models/world_model.py`
- `models/mpc_controller.py`

**Implementation:**

1. **Config & CLI** (30 min)
   ```python
   # config.py
   MODEL_CONFIG = {
       "use_ensemble": False,
       "ensemble_size": 5,
       ...
   }
   ```
   - Add `--use_ensemble` and `--ensemble_size` CLI flags

2. **Model Construction** (1 hour)
   ```python
   class EnsembleWorldModel(nn.Module):
       def forward(self, obs, action, reduce="mean"):
           # reduce="none" → [ensemble_size, batch, ...]
           # reduce="mean" → [batch, ...]
   ```

3. **Training Loop** (2 hours)
   - Bootstrap sampling for each ensemble member
   - Save as `world_model_member_{i}.pt`
   - Track per-member and aggregate losses

4. **MPC Integration** (2 hours)
   ```python
   # Risk-sensitive objective
   score = mean_return - lambda_risk * std_return
   ```
   - Query all ensemble members
   - Compute mean and variance of predictions
   - Add `lambda_risk` parameter (default 0.0)

**Acceptance Criteria:**
- [ ] Training with `--use_ensemble` produces N model files
- [ ] MPC logs include mean/variance of predicted returns
- [ ] README documents ensemble mode and risk-sensitive planning

### Task 2: Explicit Model Rollout Evaluation ✅

**Status:** COMPLETED (see `scripts/eval_world_model_rollouts.py`)

**Remaining work:**
- [ ] Add ensemble support to evaluation script
- [ ] Generate and commit reference plots to `results/`
- [ ] Update README with fidelity metrics

---

## Milestone v0.3: Latent Representation
**Goal:** Implement V-M-C architecture with explicit latent space
**Timeline:** 4-5 days
**Impact:** Foundation for visual observations and better generalization

### Task 3: Latent Layer for Vector Observations

**New files:**
- `models/encoder_decoder.py`
- `training/train_autoencoder.py`
- `models/latent_world_model.py`
- `training/train_latent_world_model.py`

**Implementation:**

1. **Encoder/Decoder Modules** (1 hour)
   ```python
   class Encoder(nn.Module):
       def __init__(self, obs_dim=9, latent_dim=16, hidden_dim=128):
           # MLP: obs → z

   class Decoder(nn.Module):
       def __init__(self, latent_dim=16, obs_dim=9, hidden_dim=128):
           # MLP: z → obs_recon
   ```

2. **Autoencoder Training** (2 hours)
   ```python
   # training/train_autoencoder.py
   z = encoder(obs_batch)
   obs_hat = decoder(z)
   loss = mse(obs_hat, obs_batch)
   ```
   - Load observations from replay buffer
   - Train/val split with early stopping
   - Save `encoder.pt`, `decoder.pt`

3. **Latent World Model** (2 hours)
   ```python
   class LatentWorldModel(nn.Module):
       # Input: [z_t, action_t]
       # Output: [z_{t+1}, r_t]
   ```
   - Pre-encode dataset: `z_t = encoder(obs_t)`
   - Train on latent transitions

4. **MPC Integration** (1 hour)
   ```python
   # If use_latent=True:
   z_0 = encoder(obs_0)
   # Roll out in latent space
   z_next, r = latent_world_model(z, a)
   ```

**Acceptance Criteria:**
- [ ] Autoencoder converges to <0.01 reconstruction error
- [ ] Latent world model matches/beats raw observation model
- [ ] MPC works seamlessly in both modes via config flag

---

## Milestone v0.4: Visual Observations
**Goal:** Demonstrate learning and control from pixels
**Timeline:** 5-6 days
**Impact:** True "visual world model" capability

### Task 4: Visual Observation Experiment

**Files to modify:**
- `envs/simple_3d_nav.py`
- `models/encoder_decoder.py` (add Conv variants)
- New: `training/train_conv_autoencoder.py`
- New: `training/train_visual_world_model.py`

**Implementation:**

1. **Visual Observations** (2 hours)
   ```python
   def get_image_obs(self):
       # Return 64x64 RGB or grayscale
       # Top-down or agent-centric view
   ```
   - Add `OBS_TYPE = "state" | "image"` config

2. **Conv Encoder/Decoder** (2 hours)
   ```python
   class ConvEncoder(nn.Module):
       # Conv layers → latent (dim 32-64)

   class ConvDecoder(nn.Module):
       # Deconv layers → reconstructed image
   ```

3. **Visual Training Pipeline** (3 hours)
   - Collect image dataset with random policy
   - Train visual autoencoder
   - Train world model on visual latents

4. **Visual MPC** (2 hours)
   ```python
   if OBS_TYPE == "image":
       z_visual = conv_encoder(image)
       # MPC in visual latent space
   ```

5. **Demo & Documentation** (1 hour)
   - Script to run visual MPC agent
   - Save rollout GIFs
   - Add "Visual World Model" README section

**Acceptance Criteria:**
- [ ] Visual AE reconstructs recognizable images
- [ ] Agent reaches goals using only visual input (>50% success)
- [ ] README includes visual results (GIF or images)

---

## Milestone v0.5: Stochastic Dynamics ✅
**Goal:** Full probabilistic world model with uncertainty quantification
**Timeline:** 3-4 days
**Impact:** State-of-the-art uncertainty handling
**Status:** COMPLETED

### Task 5: Stochastic World Model ✅

**New files:**
- `models/stochastic_world_model.py` ✅
- `models/stochastic_vae_model.py` ✅
- `models/risk_metrics.py` ✅
- `training/train_stochastic_model.py` ✅

**Implementation:**

1. **Stochastic Model** ✅
   - StochasticWorldModel with Normal distributions
   - StochasticEnsembleWorldModel combining epistemic and aleatoric uncertainty
   - StochasticVAEWorldModel with full VAE architecture

2. **Training with NLL** ✅
   - Negative log-likelihood loss implementation
   - KL divergence regularization for VAE
   - Entropy regularization to prevent collapse

3. **Probabilistic Planning** ✅
   - Risk metrics: CVaR, VaR, worst-case, mean-std, entropic
   - RiskSensitiveMPC for robust planning
   - Uncertainty propagation in trajectory rollouts

**Acceptance Criteria:**
- [x] NLL loss converges with reasonable variance values
- [x] MPC with risk penalty shows safer behavior
- [x] Documentation explains when to use stochastic vs deterministic

---

## Implementation Schedule

### Week 1
- **Day 1-2:** Complete Milestone v0.2 (Ensemble + Evaluation)
- **Day 3-5:** Complete Milestone v0.3 (Latent Representation)

### Week 2
- **Day 6-9:** Complete Milestone v0.4 (Visual Observations)
- **Day 10-11:** Complete Milestone v0.5 (Stochastic Models)
- **Day 12:** Integration testing, documentation, benchmarks

---

## GitHub Issue Templates

### Feature Issue Template
```markdown
## Feature: [Name]
**Milestone:** v0.X
**Priority:** P0/P1/P2
**Estimated:** X hours

### Context
[Why this feature matters]

### Implementation
- [ ] Step 1: [Specific file/function to modify]
- [ ] Step 2: [What to implement]
- [ ] Step 3: [How to test]

### Acceptance Criteria
- [ ] [Measurable outcome 1]
- [ ] [Measurable outcome 2]

### Files to Modify
- `path/to/file1.py`
- `path/to/file2.py`
```

### Bug Issue Template
```markdown
## Bug: [Description]
**Severity:** Critical/High/Medium/Low
**Component:** [ensemble/latent/visual/etc]

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Reproduction Steps
1. Run command: `python ...`
2. Observe error: ...

### Proposed Fix
[If known]
```

---

## Success Metrics

### v0.2 Success
- Ensemble reduces prediction uncertainty by >30%
- Model maintains <10% error for 20+ steps
- Risk-aware planning improves safety metrics

### v0.3 Success
- Latent representation compresses 9D → 16D with <1% reconstruction error
- Latent dynamics match or beat raw observation model
- Training time reduced by 20%

### v0.4 Success
- Visual control achieves >50% of state-based performance
- Image predictions accurate for 10+ frames
- End-to-end visual MPC runs at >5 Hz

### v0.5 Success
- Uncertainty estimates correlate with actual errors (ρ > 0.7)
- Stochastic model improves OOD robustness by 40%
- Risk-sensitive planning reduces failures by 25%

---

## Testing Strategy

### Unit Tests (per PR)
- Model forward passes
- Loss computations
- Data loading

### Integration Tests (per milestone)
- Full training pipeline
- MPC with trained models
- Evaluation metrics

### System Tests (before release)
- All configurations work
- Performance benchmarks
- Memory profiling

---

## Documentation Requirements

Each milestone must include:
1. Updated README section
2. Config examples
3. CLI usage examples
4. Performance metrics
5. Troubleshooting guide

---

## Notes

- Keep backward compatibility via config flags
- Benchmark performance impact of each feature
- Create git tags for each milestone
- Consider paper submission after v0.5