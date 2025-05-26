#!/usr/bin/env python3
"""
Simple wrapper script to run the sophisticated learnable scoring function training.

This script provides a simpler interface to run the training with reasonable defaults,
while still allowing customization through command line arguments.

Usage:
    python /ssd_4TB/divake/conformal-od/learnable_scoring_fn/run_training.py \
        --config_file cfg_std_rank \
        --config_path config/coco_val/

Or with default arguments:
    python /ssd_4TB/divake/conformal-od/learnable_scoring_fn/run_training.py
"""

import os
import sys
import argparse

# Add the project root to Python path
project_root = '/ssd_4TB/divake/conformal-od'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change to project directory
os.chdir(project_root)

# Import the main training function
from learnable_scoring_fn.train_scoring import main as train_main, create_parser


def create_simple_parser():
    """Create a simplified argument parser with reasonable defaults."""
    parser = argparse.ArgumentParser(
        description="Run sophisticated learnable scoring function training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Essential arguments
    parser.add_argument('--config_file', type=str, default='cfg_std_rank',
                       help='Configuration file name')
    parser.add_argument('--config_path', type=str, default='config/coco_val/',
                       help='Path to config directory')
    
    # Training parameters with good defaults
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--target_coverage', type=float, default=0.9,
                       help='Target coverage level')
    
    # Experiment setup
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Data handling
    parser.add_argument('--subset_size', type=int, default=10000,
                       help='Maximum number of training samples')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions for future use')
    parser.add_argument('--load_predictions', type=str, default=None,
                       help='Load cached predictions from this path')
    
    # Advanced options (with sophisticated defaults)
    parser.add_argument('--warmup_epochs', type=int, default=20,
                       help='Number of warmup epochs')
    parser.add_argument('--ramp_epochs', type=int, default=30,
                       help='Number of ramp epochs for lambda schedule')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--tau_update_freq', type=int, default=5,
                       help='Update tau every N epochs')
    
    # Quick options
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with reduced epochs and data')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with minimal data and epochs')
    
    return parser


def main():
    """Main wrapper function."""
    print("🚀 Starting Sophisticated Learnable Scoring Function Training")
    print("=" * 60)
    
    # Parse arguments with simplified parser
    simple_parser = create_simple_parser()
    simple_args = simple_parser.parse_args()
    
    # Create the full argument list for the main training script
    full_args = []
    
    # Add essential arguments
    full_args.extend(['--config_file', simple_args.config_file])
    full_args.extend(['--config_path', simple_args.config_path])
    
    # Add training parameters
    full_args.extend(['--num_epochs', str(simple_args.num_epochs)])
    full_args.extend(['--batch_size', str(simple_args.batch_size)])
    full_args.extend(['--learning_rate', str(simple_args.learning_rate)])
    full_args.extend(['--target_coverage', str(simple_args.target_coverage)])
    
    # Add experiment setup
    if simple_args.exp_name:
        full_args.extend(['--exp_name', simple_args.exp_name])
    full_args.extend(['--device', simple_args.device])
    full_args.extend(['--seed', str(simple_args.seed)])
    
    # Add data handling
    full_args.extend(['--subset_size', str(simple_args.subset_size)])
    # Always enable prediction saving for caching
    full_args.append('--save_predictions')
    if simple_args.load_predictions:
        full_args.extend(['--load_predictions', simple_args.load_predictions])
    
    # Add advanced options
    full_args.extend(['--warmup_epochs', str(simple_args.warmup_epochs)])
    full_args.extend(['--ramp_epochs', str(simple_args.ramp_epochs)])
    full_args.extend(['--early_stopping_patience', str(simple_args.early_stopping_patience)])
    full_args.extend(['--tau_update_freq', str(simple_args.tau_update_freq)])
    
    # Handle quick and debug modes
    if simple_args.quick:
        print("⚡ Quick mode enabled: reducing epochs and data size")
        full_args[full_args.index('--num_epochs') + 1] = '30'
        full_args[full_args.index('--subset_size') + 1] = '5000'
        full_args[full_args.index('--early_stopping_patience') + 1] = '10'
    
    if simple_args.debug:
        print("🐛 Debug mode enabled: minimal data and epochs")
        full_args[full_args.index('--num_epochs') + 1] = '5'
        full_args[full_args.index('--subset_size') + 1] = '1000'
        full_args[full_args.index('--batch_size') + 1] = '32'
        full_args[full_args.index('--early_stopping_patience') + 1] = '3'
        full_args[full_args.index('--tau_update_freq') + 1] = '2'
    
    # Add default arguments that are always needed
    full_args.extend([
        '--alpha', '0.1',
        '--label_set', 'class_threshold',
        '--label_alpha', '0.1',
        '--risk_control', 'True',
        '--save_label_set', 'True',
        '--train_frac', '0.5',
        '--cal_frac', '0.3',
        '--val_frac', '0.2',
        '--input_dim', '13',
        '--hidden_dims', '128', '64', '32',
        '--dropout_rate', '0.2',
        '--weight_decay', '0.0001',
        '--initial_lambda', '0.01',
        '--final_lambda', '0.1',
        '--margin_weight', '0.1',
        '--schedule_type', 'linear',
        '--grad_clip_norm', '1.0',
        '--output_dir', 'learnable_scoring_fn/experiments',
        '--save_every', '10'
    ])
    
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📊 Config: {simple_args.config_file} from {simple_args.config_path}")
    print(f"🎯 Target coverage: {simple_args.target_coverage}")
    print(f"📈 Epochs: {full_args[full_args.index('--num_epochs') + 1]}")
    print(f"🔧 Device: {simple_args.device}")
    print("=" * 60)
    
    # Override sys.argv with our constructed arguments
    original_argv = sys.argv.copy()
    sys.argv = ['train_scoring.py'] + full_args
    
    try:
        # Run the main training function
        exp_dir = train_main()
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        print(f"📂 Results saved to: {exp_dir}")
        print("=" * 60)
        return exp_dir
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main() 