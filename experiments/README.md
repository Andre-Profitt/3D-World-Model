# Experiments

This directory contains scripts and configurations for running experiments with the 3D World Model.

## Running Experiments

Use the `run_experiment.py` script to execute a full experiment pipeline:

```bash
python experiments/run_experiment.py --name my_experiment --mode vector --epochs 100
```

### Arguments

- `--name`: Name of the experiment (results saved to `logs/experiments/<name>`)
- `--mode`: Experiment mode (`vector`, `visual`, `latent`, `ensemble`)
- `--epochs`: Number of training epochs
- `--steps`: Number of environment steps for data collection (default: 10000)
- `--eval_episodes`: Number of evaluation episodes (default: 10)

## Experiment Modes

1.  **Vector**: Standard state-based world model.
2.  **Visual**: Visual world model using CNN encoder/decoder.
3.  **Latent**: Latent world model (V-M-C) using pre-trained autoencoder.
4.  **Ensemble**: Ensemble world model for uncertainty estimation.

## Results

Results are saved in `logs/experiments/<name>/` and include:
- `config.json`: Experiment configuration
- `metrics.json`: Evaluation metrics
- `training_log.csv`: Training loss curves
- `plots/`: Generated plots (reward, loss, etc.)
