#!/usr/bin/env python
"""
Generate commands.txt for batch experiments by reading models from base_config.yaml
"""

import argparse
import yaml
from pathlib import Path


def load_models_from_config(config_path="learnable_scoring_fn/configs/base_config.yaml"):
    """Load all model names from base_config.yaml."""
    # Check if we're in the learnable_scoring_fn directory
    if Path("configs/base_config.yaml").exists():
        config_path = "configs/base_config.yaml"
    elif not Path(config_path).exists():
        # Try absolute path
        config_path = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/configs/base_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get models from COCO dataset (since names are same for all datasets)
    models = list(config['model']['cache_dirs']['coco'].keys())
    return sorted(models)


def generate_commands(
    datasets=['coco'],
    suffix='false',
    cuda_devices=['0'],
    output_file="commands.txt",
    python_path="/home/divake/miniconda3/envs/env_cu121/bin/python"
):
    """Generate experiment commands based on parameters."""
    
    # Base command template
    base_cmd = f"{python_path} train_size_aware.py"
    
    # Load models from config
    models = load_models_from_config()
    print(f"Found {len(models)} models in base_config.yaml: {', '.join(models)}")
    
    # Generate commands
    commands = []
    
    # Distribute datasets across CUDA devices
    dataset_device_mapping = {}
    for i, dataset in enumerate(datasets):
        device_idx = i % len(cuda_devices)
        dataset_device_mapping[dataset] = cuda_devices[device_idx]
    
    # Generate commands for each combination
    for dataset in datasets:
        device = dataset_device_mapping[dataset]
        for model in models:
            cmd_parts = [
                base_cmd,
                f"--config configs/symmetric_size_aware.yaml",
                f"--base_model {model}",
                f"--dataset {dataset}",
                f"--suffix {suffix}",
                f"--device cuda:{device}"
            ]
            
            command = " ".join(cmd_parts)
            commands.append(command)
    
    # Ensure output path is in learnable_scoring_fn directory
    output_path = Path(output_file)
    if not output_path.is_absolute():
        # If relative path, put it in learnable_scoring_fn directory
        output_path = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn") / output_file
    
    # Write to file
    with open(output_path, 'w') as f:
        for cmd in commands:
            f.write(cmd + "\n")
        # Ensure file ends with newline
        if commands and not cmd.endswith('\n'):
            f.write('\n')
    
    print(f"\nGenerated {len(commands)} commands in {output_path}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models per dataset: {len(models)}")
    print(f"CUDA devices: {', '.join([f'cuda:{d}' for d in cuda_devices])}")
    
    # Show distribution
    print("\nDataset-Device mapping:")
    for dataset, device in dataset_device_mapping.items():
        print(f"  {dataset} -> cuda:{device}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment commands by reading models from base_config.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['coco'],
        choices=['coco', 'cityscapes', 'bdd100k'],
        help='Datasets to generate commands for'
    )
    
    parser.add_argument(
        '--suffix',
        default='false',
        choices=['true', 'false'],
        help='Add timestamp suffix to output directories'
    )
    
    parser.add_argument(
        '--cuda',
        nargs='+',
        default=['0'],
        help='CUDA device IDs to use (e.g., 0 1 for cuda:0 and cuda:1)'
    )
    
    parser.add_argument(
        '--output',
        default='commands.txt',
        help='Output file for commands'
    )
    
    parser.add_argument(
        '--python',
        default='/home/divake/miniconda3/envs/env_cu121/bin/python',
        help='Python executable path'
    )
    
    args = parser.parse_args()
    
    generate_commands(
        datasets=args.datasets,
        suffix=args.suffix,
        cuda_devices=args.cuda,
        output_file=args.output,
        python_path=args.python
    )


if __name__ == "__main__":
    main()