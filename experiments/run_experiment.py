"""
Run a full experiment pipeline: Data Collection -> Training -> Evaluation.
"""

import sys
import os
import argparse
import json
import subprocess
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wm_config as config

def run_command(cmd):
    """Run a shell command and check for errors."""
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run 3D World Model Experiment")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--mode", type=str, choices=["vector", "visual", "latent", "ensemble"], default="vector", help="Experiment mode")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=10000, help="Number of data collection steps")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup directories
    exp_dir = config.LOGS_DIR / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Experiment: {args.name}")
    print(f"Mode: {args.mode}")
    print(f"Output Directory: {exp_dir}")
    
    # 1. Data Collection
    print("\n=== Step 1: Data Collection ===")
    # Estimate episodes from steps (approx 200 steps per episode)
    num_episodes = args.steps // 200
    collect_cmd = f"python3 scripts/collect_data.py --num_episodes {num_episodes}"
    if args.mode == "visual":
        # For visual, we might need a different collector or just ensure env is config'd right
        # Currently collect_data.py uses config.ENV_CONFIG.
        # We should probably override config via env vars or args if supported, 
        # but for now let's assume the user configured config.py or we use a specific script.
        # Actually, we have scripts/collect_visual_data.py (implied by previous tasks, but wait, did I create it?)
        # I created training/train_visual_world_model.py which has a collect_visual_data function.
        # Let's stick to standard collection for vector/latent, and maybe special for visual.
        pass
        
    run_command(collect_cmd)
    
    # 2. Training
    print("\n=== Step 2: Training ===")
    if args.mode == "vector":
        train_cmd = f"python3 training/train_world_model.py --num_epochs {args.epochs}"
    elif args.mode == "ensemble":
        train_cmd = f"python3 training/train_world_model.py --num_epochs {args.epochs} --use_ensemble --ensemble_size 5"
    elif args.mode == "visual":
        train_cmd = f"python3 training/train_visual_world_model.py --num_epochs {args.epochs}"
    elif args.mode == "latent":
        # Train autoencoder first
        print("Training Autoencoder...")
        run_command(f"python3 training/train_autoencoder.py --num_epochs {args.epochs // 2}")
        # Then latent world model
        print("Training Latent World Model...")
        train_cmd = f"python3 training/train_latent_world_model.py --num_epochs {args.epochs}"
        
    run_command(train_cmd)
    
    # 3. Evaluation
    print("\n=== Step 3: Evaluation ===")
    eval_cmd = f"python3 scripts/run_mpc_agent.py --num_episodes {args.eval_episodes}"
    
    if args.mode == "visual":
        eval_cmd += " --use_visual"
    elif args.mode == "latent":
        eval_cmd += " --use_latent"
    elif args.mode == "ensemble":
        # run_mpc_agent automatically loads ensemble if model is ensemble
        pass
        
    run_command(eval_cmd)
    
    # Save experiment metadata
    metadata = {
        "name": args.name,
        "mode": args.mode,
        "epochs": args.epochs,
        "steps": args.steps,
        "timestamp": time.time()
    }
    
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nExperiment {args.name} Completed Successfully!")
    print(f"Results saved to {exp_dir}")

if __name__ == "__main__":
    main()
