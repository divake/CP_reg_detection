#!/usr/bin/env python3
"""
Run All Cache Generation Script
===============================

This script runs cache generation for all 13 Detectron2 models serially.
Perfect for overnight processing. Each model takes approximately 1 hour.

Usage:
    cd /ssd_4TB/divake/conformal-od
    source /home/divake/miniconda3/etc/profile.d/conda.sh
    conda activate env_cu121
    
    # Basic usage
    /home/divake/miniconda3/envs/env_cu121/bin/python scripts/run_all_cache_generation.py
    
    # Skip models that already have cache (recommended)
    /home/divake/miniconda3/envs/env_cu121/bin/python scripts/run_all_cache_generation.py --skip-existing
    
    # Run only specific models
    /home/divake/miniconda3/envs/env_cu121/bin/python scripts/run_all_cache_generation.py --models retinanet_r50,cascade_r50
    
    # Resume from a specific model
    /home/divake/miniconda3/envs/env_cu121/bin/python scripts/run_all_cache_generation.py --start-from r101fpn
    
    # Use specific GPU
    /home/divake/miniconda3/envs/env_cu121/bin/python scripts/run_all_cache_generation.py --skip-existing --gpu 0
    /home/divake/miniconda3/envs/env_cu121/bin/python scripts/run_all_cache_generation.py --skip-existing --gpu 1
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add the parent directory to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))

def check_cache_exists(cache_dir):
    """Check if cache already exists for a model."""
    cache_path = Path(cache_dir)
    required_files = [
        "predictions_train.pkl",
        "predictions_val.pkl", 
        "features_train.pt",
        "features_val.pt"
    ]
    
    if not cache_path.exists():
        return False
    
    for file_name in required_files:
        if not (cache_path / file_name).exists():
            return False
    
    return True


def run_cache_generation(model_name, model_info, python_path, script_path, 
                        max_train=None, max_val=None, gpu=None):
    """
    Run cache generation for a specific model.
    
    Args:
        model_name: Name of the model
        model_info: Model configuration dictionary
        python_path: Path to Python executable
        script_path: Path to the main cache generation script
        max_train: Maximum training images (None for all)
        max_val: Maximum validation images (None for all)
        gpu: GPU device ID to use (e.g., 0 or 1)
    
    Returns:
        tuple: (success: bool, message: str, duration: float)
    """
    print(f"\n{'='*80}")
    print(f"GENERATING CACHE FOR: {model_name}")
    print(f"{'='*80}")
    print(f"Description: {model_info['description']}")
    print(f"Checkpoint: {model_info['checkpoint_file']}")
    print(f"Cache Directory: {model_info['cache_dir']}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Build command
    # Use lower confidence threshold to ensure all models make predictions
    cmd = [
        python_path,
        script_path,
        "--model", model_name,
        "--confidence-threshold", "0.5",  # Lower threshold for better compatibility
        "--iou-threshold", "0.5"
    ]
    
    # Add limits if specified
    if max_train is not None:
        cmd.extend(["--max-train", str(max_train)])
    if max_val is not None:
        cmd.extend(["--max-val", str(max_val)])
    
    # Add GPU parameter if specified
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the cache generation
        result = subprocess.run(
            cmd,
            cwd="/ssd_4TB/divake/conformal-od",
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ SUCCESS: {model_name} cache generated successfully!")
            print(f"   Duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
            return True, f"SUCCESS - Duration: {duration/3600:.2f}h", duration
        else:
            print(f"\n❌ FAILED: {model_name} cache generation failed")
            print(f"   Return code: {result.returncode}")
            print(f"   Duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
            return False, f"FAILED - Return code: {result.returncode}, Duration: {duration/3600:.2f}h", duration
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n💥 ERROR: {model_name} failed with exception: {e}")
        return False, f"ERROR - {str(e)}, Duration: {duration/3600:.2f}h", duration


def main():
    """Main function to run cache generation for all models."""
    parser = argparse.ArgumentParser(
        description="Run cache generation for all 13 Detectron2 models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have cache generated"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of specific models to run (e.g., 'retinanet_r50,cascade_r50')"
    )
    
    parser.add_argument(
        "--start-from",
        type=str,
        help="Start from a specific model (useful for resuming)"
    )
    
    parser.add_argument(
        "--max-train",
        type=int,
        help="Maximum training images (for testing)"
    )
    
    parser.add_argument(
        "--max-val",
        type=int,
        help="Maximum validation images (for testing)"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use (e.g., 0 or 1)"
    )
    
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically answer yes to all prompts (for unattended execution)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("RUNNING CACHE GENERATION FOR ALL MODELS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
    print()
    
    # Configuration
    python_path = "/home/divake/miniconda3/envs/env_cu121/bin/python"
    script_path = "/ssd_4TB/divake/conformal-od/scripts/generate_detectron2_cache.py"
    base_cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn"
    
    # Import MODEL_REGISTRY to get all model names
    try:
        from scripts.generate_detectron2_cache import MODEL_REGISTRY
        print(f"Found {len(MODEL_REGISTRY)} models")
    except ImportError as e:
        print(f"Error importing MODEL_REGISTRY: {e}")
        return 1
    
    # Filter models based on arguments
    models_to_run = []
    
    if args.models:
        # Run specific models
        specified_models = [m.strip() for m in args.models.split(',')]
        for model_name in specified_models:
            if model_name in MODEL_REGISTRY:
                models_to_run.append((model_name, MODEL_REGISTRY[model_name]))
            else:
                print(f"Warning: Model '{model_name}' not found in registry")
    else:
        # Run all models
        models_to_run = list(MODEL_REGISTRY.items())
    
    # Handle start-from option
    if args.start_from:
        start_index = -1
        for i, (model_name, _) in enumerate(models_to_run):
            if model_name == args.start_from:
                start_index = i
                break
        
        if start_index == -1:
            print(f"Error: Start model '{args.start_from}' not found")
            return 1
        
        models_to_run = models_to_run[start_index:]
        print(f"Starting from model: {args.start_from}")
    
    # Check existing caches if requested
    if args.skip_existing:
        print("Checking for existing caches...")
        original_count = len(models_to_run)
        filtered_models = []
        
        for model_name, model_info in models_to_run:
            cache_dir = f"{base_cache_dir}/{model_info['cache_dir']}"
            if check_cache_exists(cache_dir):
                print(f"  ✅ {model_name}: Cache already exists, skipping")
            else:
                print(f"  ⏳ {model_name}: No cache found, will generate")
                filtered_models.append((model_name, model_info))
        
        models_to_run = filtered_models
        print(f"Filtered {original_count} models to {len(models_to_run)} models")
        print()
    
    if not models_to_run:
        print("No models to process!")
        return 0
    
    # Estimate total time
    estimated_time_per_model = 1.0  # hours
    total_estimated_time = len(models_to_run) * estimated_time_per_model
    estimated_completion = datetime.now() + timedelta(hours=total_estimated_time)
    
    print(f"Models to process: {len(models_to_run)}")
    print(f"Estimated time per model: {estimated_time_per_model} hours")
    print(f"Total estimated time: {total_estimated_time} hours")
    print(f"Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Confirm before starting
    if len(models_to_run) > 1:
        print("Models to process:")
        for i, (model_name, model_info) in enumerate(models_to_run, 1):
            print(f"  {i:2d}. {model_name:15} - {model_info['description']}")
        print()
        
        if not args.yes:
            try:
                response = input("Do you want to proceed? (y/N): ")
                if response.lower() != 'y':
                    print("Cancelled by user")
                    return 0
            except (EOFError, OSError):
                # Handle non-interactive environments (nohup, tmux, etc.)
                print("Non-interactive environment detected. Use --yes flag to proceed automatically.")
                print("Cancelled - add --yes flag to run unattended")
                return 1
        else:
            print("Auto-proceeding with --yes flag...")
    
    # Run cache generation for each model
    results = {}
    successful_models = []
    failed_models = []
    
    total_start_time = time.time()
    
    for i, (model_name, model_info) in enumerate(models_to_run, 1):
        print(f"\n[{i}/{len(models_to_run)}] Processing {model_name}")
        
        success, message, duration = run_cache_generation(
            model_name, model_info, python_path, script_path,
            max_train=args.max_train, max_val=args.max_val, gpu=args.gpu
        )
        
        results[model_name] = {
            'success': success,
            'message': message,
            'duration': duration,
            'description': model_info['description']
        }
        
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # Print progress
        remaining = len(models_to_run) - i
        if remaining > 0:
            avg_duration = sum(r['duration'] for r in results.values()) / len(results)
            estimated_remaining = remaining * avg_duration
            completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
            print(f"\nProgress: {i}/{len(models_to_run)} completed")
            print(f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_duration = time.time() - total_start_time
    
    # Print final summary
    print("\n" + "="*80)
    print("CACHE GENERATION SUMMARY")
    print("="*80)
    
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"Total models processed: {len(models_to_run)}")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")
    
    if successful_models:
        print(f"\n✅ SUCCESSFUL MODELS ({len(successful_models)}):")
        for model in successful_models:
            result = results[model]
            print(f"   {model:15} - {result['message']}")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
        for model in failed_models:
            result = results[model]
            print(f"   {model:15} - {result['message']}")
    
    # Save detailed results
    results_file = f"cache_generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write("Cache Generation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)\n")
        f.write(f"Successful: {len(successful_models)}/{len(models_to_run)}\n\n")
        
        for model_name, result in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            f.write(f"  Duration: {result['duration']:.2f}s ({result['duration']/3600:.2f}h)\n")
            f.write(f"  Description: {result['description']}\n")
            f.write(f"  Message: {result['message']}\n\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    print("\n" + "="*80)
    if len(failed_models) == 0:
        print("🎉 ALL CACHE GENERATION COMPLETED SUCCESSFULLY!")
        print("You can now run your analysis on all model caches.")
    else:
        print("⚠️  SOME CACHE GENERATION FAILED")
        print("Check the results above and re-run failed models if needed.")
    
    print("="*80)
    
    return 0 if len(failed_models) == 0 else 1


if __name__ == "__main__":
    exit(main()) 