#!/usr/bin/env python3
"""
Simple training script wrapper for learnable scoring function.

This script provides a convenient way to start training with sensible defaults.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run training with default parameters."""
    
    # Ensure we're in the right directory
    project_root = "/ssd_4TB/divake/conformal-od"
    os.chdir(project_root)
    
    # Default training command
    cmd = [
        sys.executable, "-m", "learnable_scoring_fn.train_scoring",
        "--config_file", "cfg_std_rank",
        "--config_path", "/ssd_4TB/divake/conformal-od/config/coco_val",
        "--subset_size", "50000",
        "--num_epochs", "100", 
        "--batch_size", "64",
        "--learning_rate", "0.001",
        "--target_coverage", "0.9",
        "--initial_lambda", "0.01",
        "--final_lambda", "0.1",
        "--warmup_epochs", "20",
        "--ramp_epochs", "30",
        "--device", "cuda",
        "--save_data",
        "--exp_name", "learnable_scoring_default"
    ]
    
    print("Starting learnable scoring function training...")
    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("- Use 50k samples from COCO validation data")
    print("- Train for 100 epochs with curriculum learning")
    print("- Target 90% coverage with adaptive lambda")
    print("- Save trained model for use in conformal prediction")
    print()
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        print("Trained model saved to: learnable_scoring_fn/experiments/learnable_scoring_default/")
        
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 