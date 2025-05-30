#!/usr/bin/env python3
"""
Run regression-based scoring function training with real data.
"""

import subprocess
import sys

def main():
    print("🚀 Starting Regression-Based Learnable Scoring Function Training")
    print("=" * 70)
    print("Using REAL model predictions from COCO validation set")
    print("=" * 70)
    
    cmd = [
        "/home/divake/miniconda3/envs/env_cu121/bin/python",
        "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/train.py",
        
        # Use the default config file
        "--learnable_config", "config/learnable_scoring_fn/default_config.yaml"
        
        # All other parameters will be loaded from the config file
        # You can still override specific parameters here if needed, e.g.:
        # "--num_epochs", "50",  # Override the config value
    ]
    
    print("📊 Using configuration from: config/learnable_scoring_fn/default_config.yaml")
    print("=" * 70)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        print("📂 Results saved to: learnable_scoring_fn/experiments/real_data_v1/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()