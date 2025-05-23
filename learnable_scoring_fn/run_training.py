#!/usr/bin/env python3
"""
Quick start script for learnable scoring function training with caching support.

This script shows how to use the caching functionality to avoid re-running 
the expensive collect_predictions phase every time.

Usage examples:

1. First run (with caching):
   python run_training.py --save_predictions

2. Subsequent runs (loading from cache):
   python run_training.py --load_predictions auto

3. With custom cache directory:
   python run_training.py --save_predictions --predictions_cache_dir /path/to/cache
"""

import subprocess
import sys
import os
from pathlib import Path

def run_training_with_cache():
    """Run training with intelligent caching."""
    
    print("Starting learnable scoring function training...")
    
    # Base command
    base_cmd = [
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
        "--exp_name", "learnable_scoring_default",
        "--alpha", "0.1",
        "--label_set", "class_threshold", 
        "--label_alpha", "0.1",
        "--risk_control", "True",
        "--save_label_set", "True"
    ]
    
    # Check if cache exists
    cache_dir = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/experiments/learnable_scoring_default/predictions_cache")
    img_list_cache = cache_dir / "coco_val_img_list.json"
    ist_list_cache = cache_dir / "coco_val_ist_list.json"
    
    # Add caching arguments based on cache availability
    if img_list_cache.exists() and ist_list_cache.exists():
        print("Found existing prediction cache - loading from cache...")
        print("This will skip the 6-10 minute prediction collection phase!")
        cmd = base_cmd + ["--load_predictions", "auto"]
    else:
        print("No cache found - will collect predictions and save cache...")
        print("This will take 6-10 minutes but will cache results for future runs.")
        cmd = base_cmd + ["--save_predictions"]
    
    print("Command:", " ".join(cmd))
    print()
    print("This will:")
    if "--load_predictions" in cmd:
        print("- Load cached predictions (fast)")
    else:
        print("- Use 50k samples from COCO validation data")
        print("- Collect predictions (6-10 minutes) and cache them")
    
    print("- Train for 100 epochs with curriculum learning")
    print("- Target 90% coverage with adaptive lambda")
    print("- Save trained model for use in conformal prediction")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(run_training_with_cache()) 