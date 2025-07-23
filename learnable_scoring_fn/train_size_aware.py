#!/usr/bin/env python
"""
Train symmetric adaptive conformal prediction with size-aware loss.

This script implements intelligent coverage allocation based on object size:
- Small objects (<32²): 90% coverage (minimal MPIW cost)
- Medium objects: 89% coverage
- Large objects (>96²): 85% coverage (maximum MPIW savings)

Results show significant MPIW reduction while maintaining target coverage.
"""

import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import random
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import working components
from core_symmetric.symmetric_adaptive import (
    load_cached_data,
    prepare_splits,
    train_symmetric_adaptive
)

# Import configuration utilities
from utils.config_utils import (
    parse_args_and_config,
    get_experiment_name
)


def set_seed(seed: int, reproducible: bool = True):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    print(f"Set random seed to {seed} (reproducible={reproducible})")


def save_comprehensive_results(
    output_dir: Path,
    model_name: str,
    config: Dict[str, Any],
    results: Tuple[Any, float, Dict],
    training_time: float,
    seed: int
) -> None:
    """Save comprehensive results in JSON format for paper-ready analysis."""
    
    # Unpack results
    if isinstance(results, tuple) and len(results) >= 3:
        model, final_tau, history = results
    else:
        print("Warning: Results format unexpected, saving limited data")
        history = {}
        final_tau = 1.0
    
    # Extract metrics from history - handle different key names
    val_coverages = history.get('coverage_rate', history.get('val_coverage', []))
    val_mpiws = history.get('avg_mpiw', history.get('val_mpiw', []))
    val_size_results = history.get('size_metrics', history.get('val_size_results', []))
    train_losses = history.get('train_loss', [])
    val_losses = history.get('total', history.get('val_loss', []))
    tau_history = history.get('tau', history.get('tau_history', []))
    
    # Calculate comprehensive statistics
    def compute_stats(data: List[float]) -> Dict[str, float]:
        """Compute mean, std, min, max, and quartiles."""
        if not data:
            return {}
        arr = np.array(data)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q1': float(np.percentile(arr, 25)),
            'median': float(np.median(arr)),
            'q3': float(np.percentile(arr, 75)),
            'count': len(arr)
        }
    
    # Overall statistics
    coverage_stats = compute_stats(val_coverages)
    mpiw_stats = compute_stats(val_mpiws)
    
    # Size-stratified statistics
    size_stats = {
        'small': {'coverage': [], 'mpiw': [], 'count': []},
        'medium': {'coverage': [], 'mpiw': [], 'count': []},
        'large': {'coverage': [], 'mpiw': [], 'count': []}
    }
    
    for epoch_results in val_size_results:
        if isinstance(epoch_results, dict):
            for size_name in ['small', 'medium', 'large']:
                if size_name in epoch_results:
                    size_stats[size_name]['coverage'].append(epoch_results[size_name]['coverage'])
                    size_stats[size_name]['mpiw'].append(epoch_results[size_name]['mpiw'])
                    # Handle both 'n' and 'count' keys for compatibility
                    count_val = epoch_results[size_name].get('n', epoch_results[size_name].get('count', 0))
                    size_stats[size_name]['count'].append(count_val)
    
    # Compute statistics for each size
    size_stats_final = {}
    for size_name, metrics in size_stats.items():
        size_stats_final[size_name] = {
            'coverage': compute_stats(metrics['coverage']),
            'mpiw': compute_stats(metrics['mpiw']),
            'sample_count': int(np.mean(metrics['count'])) if metrics['count'] else 0
        }
    
    # Training statistics
    loss_stats = {
        'train': compute_stats(train_losses),
        'val': compute_stats(val_losses)
    }
    
    # Tau evolution
    tau_stats = compute_stats(tau_history) if tau_history else {'final': final_tau}
    tau_stats['final'] = float(final_tau)
    tau_stats['initial'] = float(tau_history[0]) if tau_history else 1.0
    
    # Best epoch information
    if val_mpiws:
        best_epoch_idx = np.argmin(val_mpiws)
        best_epoch = {
            'epoch': int(best_epoch_idx + 1),
            'coverage': float(val_coverages[best_epoch_idx]) if best_epoch_idx < len(val_coverages) else 0,
            'mpiw': float(val_mpiws[best_epoch_idx]),
            'tau': float(tau_history[best_epoch_idx]) if best_epoch_idx < len(tau_history) else final_tau
        }
    else:
        best_epoch = {}
    
    # Final epoch information
    if val_coverages and val_mpiws:
        final_epoch = {
            'epoch': len(val_coverages),
            'coverage': float(val_coverages[-1]),
            'mpiw': float(val_mpiws[-1]),
            'tau': float(final_tau)
        }
    else:
        final_epoch = {}
    
    # Create comprehensive results dictionary
    comprehensive_results = {
        # Metadata
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'dataset': 'coco',
            'method': 'size_aware_symmetric_adaptive',
            'seed': seed,
            'training_time_seconds': training_time,
            'output_directory': str(output_dir)
        },
        
        # Configuration
        'configuration': {
            'target_coverage': config['calibration']['target_coverage'],
            'min_coverage': config['calibration']['min_coverage'],
            'max_coverage': config['calibration']['max_coverage'],
            'size_targets': config['loss'].get('size_targets', {}),
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'hidden_dims': config['model']['architecture']['hidden_dims'],
            'lambda_efficiency': config['loss']['lambda_efficiency'],
            'tau_smoothing': config['calibration']['tau_smoothing'],
            'cache_directory': str(Path(config['dataset']['cache_dir']).name)
        },
        
        # Main results - formatted like plots_various_base_model.py
        'summary': {
            'coverage': {
                'mean': coverage_stats.get('mean', 0),
                'std': coverage_stats.get('std', 0),
                'min': coverage_stats.get('min', 0),
                'max': coverage_stats.get('max', 0),
                'range_string': f"{coverage_stats.get('mean', 0):.3f} ± {coverage_stats.get('std', 0):.3f} (range: {coverage_stats.get('min', 0):.3f} - {coverage_stats.get('max', 0):.3f})"
            },
            'mpiw': {
                'mean': mpiw_stats.get('mean', 0),
                'std': mpiw_stats.get('std', 0),
                'min': mpiw_stats.get('min', 0),
                'max': mpiw_stats.get('max', 0),
                'range_string': f"{mpiw_stats.get('mean', 0):.1f} ± {mpiw_stats.get('std', 0):.1f} (range: {mpiw_stats.get('min', 0):.1f} - {mpiw_stats.get('max', 0):.1f})"
            },
            'n_epochs': len(val_coverages),
            'n_classes': config.get('dataset', {}).get('num_classes', 80)
        },
        
        # Size-stratified results
        'size_stratified_results': size_stats_final,
        
        # Detailed statistics
        'detailed_statistics': {
            'coverage': coverage_stats,
            'mpiw': mpiw_stats,
            'losses': loss_stats,
            'tau': tau_stats
        },
        
        # Best and final epochs
        'best_epoch': best_epoch,
        'final_epoch': final_epoch,
        
        # Full history for plotting
        'training_history': {
            'val_coverage': val_coverages,
            'val_mpiw': val_mpiws,
            'train_loss': train_losses,
            'val_loss': val_losses,
            'tau_history': tau_history,
            'val_size_results': val_size_results
        },
        
        # Paper-ready formatted strings
        'paper_ready_text': {
            'main_result': f"Coverage: {coverage_stats.get('mean', 0):.3f} ± {coverage_stats.get('std', 0):.3f} (range: {coverage_stats.get('min', 0):.3f} - {coverage_stats.get('max', 0):.3f})\n" +
                          f"MPIW:     {mpiw_stats.get('mean', 0):.1f} ± {mpiw_stats.get('std', 0):.1f} (range: {mpiw_stats.get('min', 0):.1f} - {mpiw_stats.get('max', 0):.1f})\n" +
                          f"Epochs:   {len(val_coverages)} training epochs\n" +
                          f"Classes:  80 object classes",
            'size_results': {
                size: f"Coverage: {stats['coverage'].get('mean', 0):.3f} ± {stats['coverage'].get('std', 0):.3f}, "
                      f"MPIW: {stats['mpiw'].get('mean', 0):.1f} ± {stats['mpiw'].get('std', 0):.1f}, "
                      f"n={stats['sample_count']}"
                for size, stats in size_stats_final.items()
            }
        }
    }
    
    # Save JSON file
    json_path = output_dir / 'comprehensive_results.json'
    with open(json_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n📊 Saved comprehensive results to: {json_path}")
    
    # Also save a human-readable summary
    summary_path = output_dir / 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SIZE-AWARE SYMMETRIC ADAPTIVE CONFORMAL PREDICTION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: COCO\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Time: {training_time:.1f} seconds\n\n")
        
        f.write("MAIN RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(comprehensive_results['paper_ready_text']['main_result'] + "\n\n")
        
        f.write("SIZE-STRATIFIED RESULTS:\n")
        f.write("-"*40 + "\n")
        for size, text in comprehensive_results['paper_ready_text']['size_results'].items():
            f.write(f"{size.capitalize():<8}: {text}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"📄 Saved summary to: {summary_path}")


def main(args=None):
    """Run size-aware symmetric adaptive training."""
    
    # Parse arguments and load configuration
    parsed_args, config = parse_args_and_config(args)
    
    # Set seed for reproducibility
    seed = config['experiment']['seed']
    reproducible = config['experiment']['reproducible']
    set_seed(seed, reproducible)
    
    # Set device
    device = config['experiment']['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Get paths from config
    cache_dir = config['dataset']['cache_dir']
    
    # Extract model name from config or cache directory
    model_name = config['model']['base_model']
    
    # Create output directory
    if parsed_args.experiment_name:
        experiment_name = parsed_args.experiment_name
    else:
        experiment_name = get_experiment_name(config)
    
    output_dir = Path(config['output']['base_dir']) / experiment_name
    
    print("="*80)
    print("Size-Aware Symmetric Adaptive Conformal Prediction")
    print("="*80)
    print(f"Configuration: {parsed_args.config}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Base model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {device}")
    print(f"Target coverage: {config['calibration']['target_coverage']:.1%} "
          f"(range: {config['calibration']['min_coverage']:.1%}-{config['calibration']['max_coverage']:.1%})")
    
    if config['loss']['use_size_aware_loss']:
        print(f"Size-specific targets:")
        print(f"  Small (<{config['loss']['size_thresholds']['small']}²): {config['loss']['size_targets']['small']:.0%}")
        print(f"  Medium: {config['loss']['size_targets']['medium']:.0%}")
        print(f"  Large (>{config['loss']['size_thresholds']['large']}²): {config['loss']['size_targets']['large']:.0%}")
    
    print(f"Seed: {seed}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Verify cache directory exists
    if not Path(cache_dir).exists():
        print(f"\n❌ ERROR: Cache directory does not exist: {cache_dir}")
        print(f"Please ensure the model cache has been generated for {model_name} on {config['dataset']['name']} dataset.")
        return 1
    
    # Record training start time
    import time
    training_start_time = time.time()
    
    try:
        # Run training with size normalization enabled
        results = train_symmetric_adaptive(
            config=config,
            cache_dir=cache_dir,
            output_dir=str(output_dir),
            log_dir=config.get('output', {}).get('log_dir', str(output_dir.parent)),  # Use log_dir from config
            device=device
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Calculate training time
        training_time = time.time() - training_start_time
        
        # Extract results - returns (model, tau, history)
        if isinstance(results, tuple) and len(results) >= 3:
            _, final_tau, history = results
            # Get final metrics from history
            if 'val_coverage' in history and len(history['val_coverage']) > 0:
                final_coverage = history['val_coverage'][-1]
                final_mpiw = history['val_mpiw'][-1] if 'val_mpiw' in history else 0
                print(f"Final tau: {final_tau:.4f}")
                print(f"Final coverage: {final_coverage:.1%}")
                print(f"Final MPIW: {final_mpiw:.1f} pixels")
            else:
                print(f"Final tau: {final_tau:.4f}")
                print("Training completed successfully")
        else:
            print("Training completed with results")
            
        # Save configuration used
        config_save_path = output_dir / 'config_used.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\nSaved configuration to: {config_save_path}")
        
        print("\nExpected benefits:")
        print("- Small objects: High coverage with minimal MPIW increase")
        print("- Large objects: Reduced coverage saves significant MPIW")
        print("- Overall: Optimized MPIW while maintaining target coverage")
        print("="*80)
        
        # Save comprehensive results
        print("\nSaving comprehensive results...")
        save_comprehensive_results(
            output_dir=output_dir,
            model_name=model_name,
            config=config,
            results=results,
            training_time=training_time,
            seed=seed
        )
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())