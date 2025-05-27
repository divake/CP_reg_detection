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
        
        # Model configuration
        "--config_file", "cfg_std_rank",
        "--config_path", "config/coco_val/",
        
        # Architecture
        "--hidden_dims", "256", "128", "64",
        "--dropout_rate", "0.15",
        
        # Training parameters
        "--num_epochs", "100",
        "--batch_size", "128",
        "--learning_rate", "0.001",
        "--weight_decay", "0.0001",
        
        # Loss weights (adjusted for better coverage)
        "--target_coverage", "0.9",
        "--efficiency_weight", "0.05",   # Lower weight on width minimization
        "--calibration_weight", "0.1",   # Higher weight on calibration
        
        # Other parameters
        "--tau_update_freq", "5",
        "--early_stopping_patience", "20",
        "--train_frac", "0.6",
        "--cal_frac", "0.2",
        
        # Output
        "--output_dir", "learnable_scoring_fn/experiments/real_data_v1"
    ]
    
    print("📊 Configuration:")
    print(f"  • Architecture: [256, 128, 64] with 15% dropout")
    print(f"  • Learning rate: 0.001 with cosine annealing")
    print(f"  • Target coverage: 90%")
    print(f"  • Data split: 60% train, 20% cal, 20% val")
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