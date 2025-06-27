"""
Verify OAGA results with the successful configuration from attempt 3
"""
import os
import sys
import json
import torch
import numpy as np

# Add parent directory to path
sys.path.append('/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/frameworks/OAGA')

# Import the training script
from train_oaga import OAGATrainer, run_oaga_training

def verify_oaga():
    """Run OAGA with the successful configuration"""
    # Use the exact configuration from attempt 3
    config = {
        'hidden_dims': [256, 128, 64],
        'dropout': 0.1,
        'activation': 'elu',
        'learning_rate': 0.0001,
        'batch_size': 256,
        'lambda_coverage': 1.0,
        'lambda_efficiency': 0.004,
        'lambda_ordering': 0.4,
        'lambda_asymmetry': 0.1,
        'symmetric_init': True,
        'asymmetry_warmup_epochs': 10,
        'scheduler': 'cosine_warm_restarts',
        'epochs': 100,
        'patience': 20,
        'target_coverage': 0.9,
        'initial_tau': 2.0,
        'min_tau': 0.5,
        'max_tau': 10.0
    }
    
    print("Running OAGA verification with successful configuration...")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run training
    results = run_oaga_training(config, attempt_number=6)
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS:")
    print("="*80)
    print(f"Coverage: {results['best_coverage']:.4f} (Target: 0.8877)")
    print(f"MPIW: {results['best_mpiw']:.2f} (Target: 41.92)")
    print(f"MPIW Reduction: {results['mpiw_reduction']:.2f}% (Target: 13.57%)")
    print("="*80)
    
    return results

if __name__ == "__main__":
    verify_oaga()