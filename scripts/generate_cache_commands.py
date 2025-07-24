#!/usr/bin/env python
"""
Generate cache_commands.txt for batch cache generation from detectron2 models.
"""

import argparse
from pathlib import Path


def get_model_names(checkpoint_dir="/ssd_4TB/divake/conformal-od/checkpoints_variants"):
    """Get all model names from checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    model_files = list(checkpoint_path.glob("*.pkl"))
    
    # Extract model names (remove .pkl extension)
    model_names = [f.stem for f in model_files]
    return sorted(model_names)


def generate_commands(
    datasets=['coco'],
    cuda_devices=['0'],
    checkpoint_dir="/ssd_4TB/divake/conformal-od/checkpoints_variants",
    output_file="cache_commands.txt",
    python_path="/home/divake/miniconda3/envs/env_cu121/bin/python"
):
    """Generate cache generation commands."""
    
    # Base command template
    base_cmd = f"{python_path} generate_detectron2_cache.py"
    
    # Get all model names
    model_names = get_model_names(checkpoint_dir)
    print(f"Found {len(model_names)} models in {checkpoint_dir}")
    print(f"Models: {', '.join(model_names[:5])}..." if len(model_names) > 5 else f"Models: {', '.join(model_names)}")
    
    # Generate commands
    commands = []
    
    # Distribute models across CUDA devices
    model_device_mapping = {}
    for i, model in enumerate(model_names):
        device_idx = i % len(cuda_devices)
        model_device_mapping[model] = cuda_devices[device_idx]
    
    for dataset in datasets:
        for model in model_names:
            device = model_device_mapping[model]
            cmd_parts = [
                base_cmd,
                f"--model {model}",
                f"--dataset {dataset}",
                f"--gpu {device}"
            ]
            
            command = " ".join(cmd_parts)
            commands.append(command)
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for cmd in commands:
            f.write(cmd + "\n")
    
    print(f"\nGenerated {len(commands)} commands in {output_path}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models per dataset: {len(model_names)}")
    print(f"CUDA devices: {', '.join([f'cuda:{d}' for d in cuda_devices])}")
    
    # Show distribution
    print("\nModel-Device distribution:")
    for model, device in sorted(model_device_mapping.items())[:5]:
        print(f"  {model} -> cuda:{device}")
    if len(model_device_mapping) > 5:
        print(f"  ... and {len(model_device_mapping) - 5} more models")
    
    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Generate cache generation commands for detectron2 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['bdd100k', 'cityscapes'],
        choices=['coco', 'cityscapes', 'bdd100k'],
        help='Datasets to generate commands for'
    )
    
    parser.add_argument(
        '--cuda',
        nargs='+',
        default=['0'],
        help='CUDA device IDs to use (e.g., 0 1 for cuda:0 and cuda:1)'
    )
    
    parser.add_argument(
        '--output',
        default='cache_commands.txt',
        help='Output file for commands'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        default='/ssd_4TB/divake/conformal-od/checkpoints_variants',
        help='Directory containing model checkpoints'
    )
    
    parser.add_argument(
        '--python',
        default='/home/divake/miniconda3/envs/env_cu121/bin/python',
        help='Python executable path'
    )
    
    args = parser.parse_args()
    
    generate_commands(
        datasets=args.datasets,
        cuda_devices=args.cuda,
        checkpoint_dir=args.checkpoint_dir,
        output_file=args.output,
        python_path=args.python
    )


if __name__ == "__main__":
    main()