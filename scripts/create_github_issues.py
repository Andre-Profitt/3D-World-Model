#!/usr/bin/env python3
"""
Generate GitHub issues from ROADMAP.md milestones.

This script creates draft issues that can be reviewed and posted to GitHub.
"""

import json
from pathlib import Path
from typing import Dict, List


# Milestone v0.2 Issues
MILESTONE_V02_ISSUES = [
    {
        "title": "[TASK] Wire Ensemble World Model End-to-End",
        "body": """## Task: Wire Ensemble World Model End-to-End

**Milestone:** v0.2
**Section:** Uncertainty & Evaluation
**Estimated Time:** 8 hours
**Priority:** P0

### Specification
Make the ensemble option fully functional and integrated into both training and MPC, enabling uncertainty-aware world models.

### Implementation Checklist

#### Code Changes
- [ ] `config.py`: Add `use_ensemble` and `ensemble_size` parameters
- [ ] `models/world_model.py`: Implement `EnsembleWorldModel` class
- [ ] `training/train_world_model.py`: Add ensemble training with bootstrap sampling
- [ ] `models/mpc_controller.py`: Update to handle ensemble predictions
- [ ] `scripts/run_mpc_agent.py`: Support loading ensemble checkpoints

#### Specific Work Items

1. **Config & CLI** (30 min)
   - [ ] Add `MODEL_CONFIG["use_ensemble"]: bool`
   - [ ] Add `MODEL_CONFIG["ensemble_size"]: int`
   - [ ] Add CLI flags `--use_ensemble` and `--ensemble_size`

2. **Model Construction** (1 hour)
   - [ ] Create `EnsembleWorldModel` wrapper class
   - [ ] Implement `forward(obs, action, reduce="mean"|"none")`
   - [ ] Handle ensemble member initialization

3. **Training Loop** (2 hours)
   - [ ] Implement bootstrap sampling for each member
   - [ ] Save checkpoints as `world_model_member_{i}.pt`
   - [ ] Track per-member and aggregate losses

4. **MPC Integration** (2 hours)
   - [ ] Query all ensemble members during rollout
   - [ ] Compute mean and variance of predictions
   - [ ] Implement risk-sensitive objective: `score = mean_return - lambda_risk * std_return`
   - [ ] Add `lambda_risk` to MPC_CONFIG

#### Tests
- [ ] Unit test for `EnsembleWorldModel.forward()`
- [ ] Test bootstrap sampling produces different datasets
- [ ] Test risk-sensitive planning changes behavior
- [ ] Verify ensemble reduces prediction variance

#### Documentation
- [ ] Add ensemble explanation to README
- [ ] Document risk-sensitive planning usage
- [ ] Add config example for ensemble mode

### Definition of Done
- [ ] Training with `--use_ensemble True --ensemble_size 5` produces 5 model files
- [ ] MPC logs include mean and variance of predicted returns
- [ ] Risk-sensitive planning (`lambda_risk > 0`) shows more conservative behavior
- [ ] README updated with ensemble documentation
""",
        "labels": ["task", "enhancement", "v0.2"]
    },
    {
        "title": "[TASK] Add Ensemble Support to Evaluation Script",
        "body": """## Task: Add Ensemble Support to Model Evaluation

**Milestone:** v0.2
**Section:** Uncertainty & Evaluation
**Estimated Time:** 2 hours
**Priority:** P1

### Specification
Extend `scripts/eval_world_model_rollouts.py` to support ensemble models and uncertainty metrics.

### Implementation Checklist

#### Code Changes
- [ ] Load ensemble checkpoints when `use_ensemble=True`
- [ ] Compute prediction mean and variance across ensemble
- [ ] Add uncertainty correlation metrics
- [ ] Plot uncertainty vs actual error

#### Specific Work Items
- [ ] Modify model loading to handle multiple checkpoints
- [ ] Add `--use_ensemble` flag to evaluation script
- [ ] Compute epistemic uncertainty from ensemble variance
- [ ] Generate plots showing uncertainty bands
- [ ] Add correlation analysis between uncertainty and error

#### Tests
- [ ] Verify ensemble evaluation produces uncertainty metrics
- [ ] Check plots include confidence intervals
- [ ] Ensure single model mode still works

### Definition of Done
- [ ] Evaluation script works with both single and ensemble models
- [ ] Plots show prediction uncertainty bands
- [ ] Uncertainty correlates with actual errors (ρ > 0.5)
""",
        "labels": ["task", "enhancement", "v0.2", "evaluation"]
    }
]

# Milestone v0.3 Issues
MILESTONE_V03_ISSUES = [
    {
        "title": "[TASK] Implement Encoder/Decoder Modules",
        "body": """## Task: Implement Encoder/Decoder for Latent Representation

**Milestone:** v0.3
**Section:** Latent Representation
**Estimated Time:** 3 hours
**Priority:** P0

### Specification
Create encoder/decoder modules to map observations to/from latent space.

### Implementation Checklist

#### Code Changes
- [ ] Create `models/encoder_decoder.py`
- [ ] Implement `Encoder` class (MLP: obs_dim → latent_dim)
- [ ] Implement `Decoder` class (MLP: latent_dim → obs_dim)
- [ ] Add latent configuration to `config.py`

#### Specific Implementation
```python
class Encoder(nn.Module):
    def __init__(self, obs_dim=9, latent_dim=16, hidden_dim=128):
        # 2-3 layer MLP with ReLU activations

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, obs_dim=9, hidden_dim=128):
        # Mirror architecture of encoder
```

#### Tests
- [ ] Test encoder output shape matches latent_dim
- [ ] Test decoder reconstruction shape matches obs_dim
- [ ] Verify gradient flow through both modules

### Definition of Done
- [ ] Modules instantiate without errors
- [ ] Forward pass produces correct shapes
- [ ] Can be imported from models package
""",
        "labels": ["task", "enhancement", "v0.3", "architecture"]
    },
    {
        "title": "[TASK] Create Autoencoder Training Script",
        "body": """## Task: Implement Autoencoder Training Pipeline

**Milestone:** v0.3
**Section:** Latent Representation
**Estimated Time:** 4 hours
**Priority:** P0
**Blocked By:** Encoder/Decoder implementation

### Specification
Create training script for observation autoencoder with proper evaluation.

### Implementation Checklist

#### Code Changes
- [ ] Create `training/train_autoencoder.py`
- [ ] Load observations from replay buffer
- [ ] Implement reconstruction loss (MSE)
- [ ] Add train/val split (90/10)
- [ ] Implement early stopping
- [ ] Save encoder.pt and decoder.pt separately

#### Training Features
- [ ] Learning rate scheduling
- [ ] Gradient clipping
- [ ] TensorBoard logging
- [ ] Checkpoint best model based on val loss

#### Evaluation Metrics
- [ ] Track reconstruction MSE
- [ ] Log example reconstructions
- [ ] Compute latent space statistics

### Definition of Done
- [ ] Training converges to <0.01 reconstruction error
- [ ] Validation loss stops improving (early stopping works)
- [ ] Saved models can be loaded and used
- [ ] Training completes in <10 minutes for 1M observations
""",
        "labels": ["task", "enhancement", "v0.3", "training"]
    }
]

# Milestone v0.4 Issues
MILESTONE_V04_ISSUES = [
    {
        "title": "[TASK] Add Visual Observations to Environment",
        "body": """## Task: Implement Camera/Image Observations

**Milestone:** v0.4
**Section:** Visual Observations
**Estimated Time:** 4 hours
**Priority:** P0

### Specification
Add camera rendering to produce image observations from the 3D environment.

### Implementation Checklist

#### Code Changes
- [ ] `envs/simple_3d_nav.py`: Add `get_image_obs()` method
- [ ] Implement top-down camera view (64x64)
- [ ] Implement agent-centric camera view
- [ ] Add `OBS_TYPE` config option
- [ ] Support grayscale and RGB modes

#### Camera Implementation
- [ ] Use matplotlib for rendering (no external deps)
- [ ] Render agent as circle/square
- [ ] Render goal as star/target
- [ ] Show boundaries as lines
- [ ] Add optional trajectory trail

#### Config Updates
```python
OBS_CONFIG = {
    "type": "state" | "image" | "state+image",
    "image_size": (64, 64),
    "image_channels": 1 | 3,
    "camera_mode": "top_down" | "agent_centric"
}
```

### Definition of Done
- [ ] Environment can return image observations
- [ ] Images clearly show agent, goal, and boundaries
- [ ] Both camera modes work correctly
- [ ] No significant performance degradation
""",
        "labels": ["task", "enhancement", "v0.4", "environment"]
    }
]

# Milestone v0.5 Issues
MILESTONE_V05_ISSUES = [
    {
        "title": "[TASK] Implement Stochastic World Model",
        "body": """## Task: Create Probabilistic Dynamics Model

**Milestone:** v0.5
**Section:** Stochastic Dynamics
**Estimated Time:** 6 hours
**Priority:** P0

### Specification
Implement world model that outputs distributions over next states and rewards.

### Implementation Checklist

#### Code Changes
- [ ] Create `models/stochastic_world_model.py`
- [ ] Output μ and log(σ²) for latent states
- [ ] Output μ and log(σ²) for rewards
- [ ] Implement reparameterization trick
- [ ] Add NLL loss function

#### Model Architecture
```python
class StochasticLatentWorldModel(nn.Module):
    def forward(self, z, a):
        # Output size: 2 * latent_dim + 2 (mean and logvar for z and r)
        # Returns: mu_z, logvar_z, mu_r, logvar_r
```

#### Training Updates
- [ ] Modify training script for `--stochastic` mode
- [ ] Implement ELBO loss (reconstruction + KL)
- [ ] Add KL annealing schedule
- [ ] Track uncertainty metrics during training

#### MPC Integration
- [ ] Sample trajectories from distributions
- [ ] Propagate uncertainty through planning
- [ ] Implement risk metrics (CVaR, worst-case)

### Definition of Done
- [ ] Model trains with stable NLL loss
- [ ] Uncertainty estimates correlate with errors (ρ > 0.7)
- [ ] Risk-sensitive planning shows measurably safer behavior
- [ ] No mode collapse (reasonable variance values)
""",
        "labels": ["task", "enhancement", "v0.5", "uncertainty"]
    }
]


def generate_issue_files():
    """Generate issue files that can be posted to GitHub."""

    issues_dir = Path(".github/issues")
    issues_dir.mkdir(exist_ok=True, parents=True)

    all_issues = [
        ("v0.2", MILESTONE_V02_ISSUES),
        ("v0.3", MILESTONE_V03_ISSUES),
        ("v0.4", MILESTONE_V04_ISSUES),
        ("v0.5", MILESTONE_V05_ISSUES),
    ]

    issue_commands = []

    for milestone, issues in all_issues:
        for i, issue in enumerate(issues):
            # Create issue file
            title_slug = issue['title'].replace('[TASK] ', '').replace(' ', '_').replace('/', '_').lower()
            title_slug = ''.join(c for c in title_slug if c.isalnum() or c == '_')[:30]
            filename = f"{milestone}_{i+1:02d}_{title_slug}.md"
            filepath = issues_dir / filename

            with open(filepath, 'w') as f:
                f.write(f"# {issue['title']}\n\n")
                f.write(f"Labels: {', '.join(issue['labels'])}\n\n")
                f.write("---\n\n")
                f.write(issue['body'])

            # Create gh command
            labels_str = ','.join(issue['labels'])
            cmd = f'gh issue create --title "{issue["title"]}" --body-file .github/issues/{filename} --label "{labels_str}"'
            issue_commands.append(cmd)

    # Save commands to script
    script_path = issues_dir / "create_all_issues.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to create all GitHub issues\n")
        f.write("# Run with: bash .github/issues/create_all_issues.sh\n\n")

        for cmd in issue_commands:
            f.write(f"{cmd}\n")
            f.write("sleep 2  # Avoid rate limiting\n\n")

    script_path.chmod(0o755)

    print(f"Generated {len(issue_commands)} issue files in .github/issues/")
    print(f"To create issues on GitHub, run:")
    print(f"  bash {script_path}")

    return issues_dir


def print_summary():
    """Print summary of roadmap milestones."""

    print("\n" + "="*60)
    print("3D World Model - Development Roadmap Summary")
    print("="*60)

    milestones = [
        ("v0.2", "Uncertainty & Evaluation", "3-4 days", 2),
        ("v0.3", "Latent Representation", "4-5 days", 2),
        ("v0.4", "Visual Observations", "5-6 days", 1),
        ("v0.5", "Stochastic Dynamics", "3-4 days", 1),
    ]

    total_issues = sum(count for _, _, _, count in milestones)

    print(f"\nTotal Milestones: {len(milestones)}")
    print(f"Total Issues: {total_issues}")
    print(f"Estimated Timeline: 2-3 weeks")

    print("\n" + "-"*40)
    print("Milestones:")
    print("-"*40)

    for version, name, timeline, issue_count in milestones:
        print(f"{version}: {name}")
        print(f"  Timeline: {timeline}")
        print(f"  Issues: {issue_count}")

    print("\n" + "-"*40)
    print("Key Deliverables:")
    print("-"*40)
    print("- Ensemble world model with uncertainty")
    print("- Latent space dynamics")
    print("- Visual observation support")
    print("- Stochastic predictions")
    print("- Risk-aware planning")
    print("- Comprehensive evaluation metrics")


if __name__ == "__main__":
    print("Generating GitHub issues from ROADMAP.md...")
    issues_dir = generate_issue_files()
    print_summary()

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Review generated issues in .github/issues/")
    print("2. Edit any issues as needed")
    print("3. Run the creation script to post to GitHub")
    print("4. Assign issues to contributors")
    print("5. Start with v0.2 milestone (highest priority)")