"""Configuration utilities for loading and merging YAML configs."""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import copy
from datetime import datetime


def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Args:
        base_dict: Base configuration dictionary
        override_dict: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base_dict)
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_config(config_path: Path, base_config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, with support for inheritance.
    
    Args:
        config_path: Path to the configuration file
        base_config_dir: Directory to search for base configs (if not specified, uses config_path's parent)
        
    Returns:
        Loaded configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for base config inheritance
    if 'base_config' in config:
        base_config_name = config.pop('base_config')
        
        # Determine base config directory
        if base_config_dir is None:
            base_config_dir = config_path.parent
            
        base_config_path = base_config_dir / base_config_name
        
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")
            
        # Load base config recursively
        base_config = load_config(base_config_path, base_config_dir)
        
        # Merge configs
        config = deep_merge(base_config, config)
    
    return config


def apply_model_cache_dir(config: Dict[str, Any], base_model: str) -> Dict[str, Any]:
    """
    Apply the appropriate cache directory based on the selected dataset and model.
    
    Args:
        config: Configuration dictionary
        base_model: Name of the base model
        
    Returns:
        Updated configuration with correct cache directory
        
    Raises:
        ValueError: If model not found in cache directories
    """
    # Get dataset name
    dataset_name = config.get('dataset', {}).get('name', 'coco')
    
    # Check if model has a cache directory mapping
    if 'model' in config and 'cache_dirs' in config['model']:
        cache_dirs = config['model']['cache_dirs']
        
        # Check if dataset exists in cache_dirs
        if dataset_name in cache_dirs:
            dataset_cache_dirs = cache_dirs[dataset_name]
            
            # Check if model exists for this dataset
            if base_model in dataset_cache_dirs:
                # Set the cache directory for this model and dataset
                if 'dataset' not in config:
                    config['dataset'] = {}
                config['dataset']['cache_dir'] = dataset_cache_dirs[base_model]
                print(f"✓ Model found: {base_model}")
                print(f"✓ Cache directory: {dataset_cache_dirs[base_model]}")
            else:
                available_models = list(dataset_cache_dirs.keys())
                error_msg = (
                    f"\n❌ ERROR: Model '{base_model}' not found in cache directories for dataset '{dataset_name}'\n"
                    f"Available models for {dataset_name}: {', '.join(sorted(available_models))}\n"
                    f"Please add the cache directory for '{base_model}' in base_config.yaml or use one of the available models."
                )
                raise ValueError(error_msg)
        else:
            available_datasets = list(cache_dirs.keys())
            error_msg = (
                f"\n❌ ERROR: Dataset '{dataset_name}' not found in cache directories\n"
                f"Available datasets: {', '.join(sorted(available_datasets))}\n"
                f"Please add cache directories for '{dataset_name}' in base_config.yaml."
            )
            raise ValueError(error_msg)
    else:
        raise ValueError("No cache_dirs configuration found in model config")
    
    return config


def resolve_paths(config: Dict[str, Any], base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in config to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for resolving relative paths
        
    Returns:
        Config with resolved paths
    """
    if base_path is None:
        base_path = Path.cwd()
    
    def _resolve_value(value):
        if isinstance(value, str) and ('/' in value or '\\' in value):
            path = Path(value)
            if not path.is_absolute():
                return str(base_path / path)
        return value
    
    def _resolve_dict(d):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = _resolve_dict(value)
            elif isinstance(value, list):
                result[key] = [_resolve_value(v) if isinstance(v, str) else v for v in value]
            else:
                result[key] = _resolve_value(value)
        return result
    
    return _resolve_dict(config)


def override_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values with command-line arguments.
    
    Args:
        config: Base configuration
        overrides: Dictionary of overrides from command line
        
    Returns:
        Updated configuration
    """
    result = copy.deepcopy(config)
    
    for key, value in overrides.items():
        if value is not None:
            # Handle nested keys (e.g., "model.architecture.hidden_dims")
            keys = key.split('.')
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Convert string representations of lists/dicts
            if isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = eval(value)
                    except:
                        pass
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.replace('.', '').replace('-', '').isdigit():
                    value = float(value) if '.' in value else int(value)
            
            current[keys[-1]] = value
    
    return result


def get_experiment_name(config: Dict[str, Any], add_timestamp: bool = True, prefix: Optional[str] = None) -> str:
    """
    Generate experiment name based on configuration.
    
    Args:
        config: Configuration dictionary
        add_timestamp: Whether to add timestamp suffix
        prefix: Optional prefix for experiment name
        
    Returns:
        Experiment name string
    """
    # Extract key identifiers
    dataset = config.get('dataset', {}).get('name', 'unknown')
    model = config.get('model', {}).get('base_model', 'unknown')
    
    # Build name parts
    parts = []
    if prefix:
        parts.append(prefix)
    parts.extend([dataset, model])
    
    # Add timestamp if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    return "_".join(parts)


def create_argparser() -> argparse.ArgumentParser:
    """
    Create argument parser with common configuration options.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train symmetric adaptive conformal prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='configs/symmetric_size_aware.yaml',
        help='Path to configuration file'
    )
    
    # Dataset options
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['coco', 'cityscapes', 'bdd100k'],
        help='Dataset name (REQUIRED - no default dataset)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        help='Cache directory for features/predictions'
    )
    
    # Model options
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='Base model architecture (REQUIRED - no default model)'
    )
    parser.add_argument(
        '--hidden_dims',
        type=str,
        help='Hidden dimensions as list, e.g., "[256,128,64]"'
    )
    
    # Training options
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cuda:0', 'cuda:1', 'cpu'],
        help='Device to use for training'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for models and results'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Experiment name (auto-generated if not provided)'
    )
    
    # Loss options
    parser.add_argument(
        '--use_size_aware',
        action='store_true',
        help='Use size-aware loss function'
    )
    parser.add_argument(
        '--lambda_efficiency',
        type=float,
        help='Efficiency loss weight'
    )
    
    # Calibration options
    parser.add_argument(
        '--target_coverage',
        type=float,
        help='Target coverage rate'
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Add timestamp suffix to output directory (default: true)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    return parser


def parse_args_and_config(args=None):
    """
    Parse command-line arguments and load configuration.
    
    Args:
        args: Optional list of arguments (for testing)
        
    Returns:
        Tuple of (parsed_args, merged_config)
    """
    parser = create_argparser()
    parsed_args = parser.parse_args(args)
    
    # Load config file
    config_path = Path(parsed_args.config)
    if not config_path.is_absolute():
        # Try relative to script location
        config_path = Path(__file__).parent.parent / parsed_args.config
    
    config = load_config(config_path)
    
    # Build overrides from command line
    overrides = {}
    
    # Direct mappings
    if parsed_args.dataset:
        overrides['dataset.name'] = parsed_args.dataset
    if parsed_args.cache_dir:
        overrides['dataset.cache_dir'] = parsed_args.cache_dir
    if parsed_args.base_model:
        overrides['model.base_model'] = parsed_args.base_model
    if parsed_args.hidden_dims:
        overrides['model.architecture.hidden_dims'] = parsed_args.hidden_dims
    if parsed_args.epochs:
        overrides['training.epochs'] = parsed_args.epochs
    if parsed_args.batch_size:
        overrides['training.batch_size'] = parsed_args.batch_size
    if parsed_args.learning_rate:
        overrides['training.learning_rate'] = parsed_args.learning_rate
    if parsed_args.device:
        overrides['experiment.device'] = parsed_args.device
    if parsed_args.output_dir:
        overrides['output.base_dir'] = parsed_args.output_dir
    if parsed_args.use_size_aware:
        overrides['loss.use_size_aware_loss'] = True
    if parsed_args.lambda_efficiency:
        overrides['loss.lambda_efficiency'] = parsed_args.lambda_efficiency
    if parsed_args.target_coverage:
        overrides['calibration.target_coverage'] = parsed_args.target_coverage
    if parsed_args.seed:
        overrides['experiment.seed'] = parsed_args.seed
    if parsed_args.debug:
        overrides['logging.level'] = 'DEBUG'
    
    # Apply overrides
    config = override_config(config, overrides)
    
    # Get dataset name - must be specified
    dataset_name = config.get('dataset', {}).get('name')
    if not dataset_name:
        raise ValueError(
            "\n❌ ERROR: No dataset specified!\n"
            "You must specify a dataset using --dataset argument.\n"
            "Example: python train_size_aware.py --dataset coco --base_model resnet101"
        )
    
    # Set num_classes based on dataset
    dataset_num_classes = {
        'coco': 80,
        'cityscapes': 8,
        'bdd100k': 10
    }
    if dataset_name in dataset_num_classes:
        config['dataset']['num_classes'] = dataset_num_classes[dataset_name]
    
    # Get model name - must be specified
    model_name = config.get('model', {}).get('base_model')
    if not model_name:
        raise ValueError(
            "\n❌ ERROR: No base model specified!\n"
            "You must specify a model using --base_model argument.\n"
            "Example: python train_size_aware.py --dataset coco --base_model resnet101"
        )
    
    # Apply model-specific cache directory
    config = apply_model_cache_dir(config, model_name)
    
    # Resolve paths
    config = resolve_paths(config)
    
    # Generate experiment name if not provided
    if not parsed_args.experiment_name:
        # Use suffix flag to determine if timestamp should be added
        add_timestamp = parsed_args.suffix.lower() == 'true'
        parsed_args.experiment_name = get_experiment_name(config, add_timestamp=add_timestamp)
    
    return parsed_args, config