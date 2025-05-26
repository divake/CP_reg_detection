#!/usr/bin/env python3
"""
Main training script for learnable scoring function.

This script implements the training loop following the classification framework pattern:
1. Calibration phase: Calculate tau using calibration set
2. Training phase: Train scoring function with fixed tau  
3. Validation phase: Evaluate with learned scores and current tau
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
import logging
import numpy as np

# Add project paths - ensure conformal-od is first in path
project_root = "/ssd_4TB/divake/conformal-od"
detectron2_path = "/ssd_4TB/divake/conformal-od/detectron2"

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if detectron2_path not in sys.path:
    sys.path.insert(0, detectron2_path)

# Try different import paths for detectron2 compatibility
try:
    from detectron2.data import get_detection_dataset_dicts
except ImportError:
    try:
        from detectron2.data.datasets import get_detection_dataset_dicts
    except ImportError:
        from detectron2.data.catalog import get_detection_dataset_dicts

# Local imports - with error handling
try:
    from .model import ScoringMLP, CoverageLoss, AdaptiveLambdaScheduler, save_model, load_model
    from .feature_utils import FeatureExtractor, get_feature_names
    from .data_utils import prepare_training_data, split_data, COCOClassMapper
except ImportError as e:
    print(f"Warning: Local import failed: {e}")
    # Try absolute imports as fallback
    from learnable_scoring_fn.model import ScoringMLP, CoverageLoss, AdaptiveLambdaScheduler, save_model, load_model
    from learnable_scoring_fn.feature_utils import FeatureExtractor, get_feature_names
    from learnable_scoring_fn.data_utils import prepare_training_data, split_data, COCOClassMapper

from data import data_loader
from util import io_file
from control.collect_predictions import collect_predictions


def create_parser():
    """Create argument parser for training."""
    parser = argparse.ArgumentParser(description="Train learnable scoring function for conformal prediction")
    
    # Data and model configuration
    parser.add_argument('--config_file', type=str, required=True, help='Configuration file name')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config directory')
    
    # Caching options for collect_predictions
    parser.add_argument('--load_predictions', type=str, default=None, help='Load cached predictions from this path (skip collect_predictions)')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions for future use')
    parser.add_argument('--predictions_cache_dir', type=str, default=None, help='Directory to save/load predictions cache')
    
    # Training data configuration
    parser.add_argument('--subset_size', type=int, default=10000, help='Maximum number of training samples')
    parser.add_argument('--train_frac', type=float, default=0.5, help='Fraction of data for training')
    parser.add_argument('--cal_frac', type=float, default=0.3, help='Fraction of data for calibration')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Fraction of data for validation')
    
    # Model architecture
    parser.add_argument('--input_dim', type=int, default=13, help='Input feature dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32], help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Coverage loss parameters
    parser.add_argument('--target_coverage', type=float, default=0.9, help='Target coverage level')
    parser.add_argument('--initial_lambda', type=float, default=0.01, help='Initial lambda for coverage loss')
    parser.add_argument('--final_lambda', type=float, default=0.1, help='Final lambda for coverage loss')
    parser.add_argument('--margin_weight', type=float, default=0.1, help='Weight for margin-based loss')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Number of warmup epochs')
    parser.add_argument('--ramp_epochs', type=int, default=30, help='Number of ramp epochs for lambda schedule')
    parser.add_argument('--schedule_type', type=str, default='linear', help='Lambda schedule type')
    
    # Advanced training parameters
    parser.add_argument('--tau_update_freq', type=int, default=5, help='Update tau every N epochs')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='learnable_scoring_fn/experiments', help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Data saving/loading
    parser.add_argument('--save_data', action='store_true', help='Save processed training data')
    parser.add_argument('--load_data', type=str, default=None, help='Load preprocessed data from path')
    
    # Required arguments for compatibility
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha level for conformal prediction')
    parser.add_argument('--label_set', type=str, default='class_threshold', help='Label set type')
    parser.add_argument('--label_alpha', type=float, default=0.1, help='Label alpha')
    parser.add_argument('--risk_control', type=bool, default=True, help='Risk control flag')
    parser.add_argument('--save_label_set', type=bool, default=True, help='Save label set flag')
    
    return parser


def calculate_sophisticated_tau(scoring_model: ScoringMLP, cal_data: dict, 
                               feature_extractor: FeatureExtractor,
                               alpha: float = 0.1, device: torch.device = None, 
                               logger=None) -> torch.Tensor:
    """
    Calculate tau using sophisticated approach following classification framework.
    
    Uses the full calibration set to compute tau as the (1-alpha) quantile
    of learned nonconformity scores, with advanced handling for stability.
    
    Args:
        scoring_model: Trained scoring model
        cal_data: Calibration data dictionary
        feature_extractor: Feature extractor
        alpha: Miscoverage level (1-alpha = coverage)
        device: Device to use
        logger: Logger instance
        
    Returns:
        tau: Single tau threshold value
    """
    scoring_model.eval()
    
    with torch.no_grad():
        # Extract and normalize features
        cal_features = feature_extractor.normalize_features(cal_data['features'].to(device))
        
        if logger is not None:
            logger.info(f"Cal features shape: {cal_features.shape}")
        
        # Validate input dimensions
        expected_input_dim = scoring_model.input_dim
        actual_input_dim = cal_features.shape[1]
        if actual_input_dim != expected_input_dim:
            error_msg = f"Feature dimension mismatch: model expects {expected_input_dim} but got {actual_input_dim}"
            if logger is not None:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Get learned nonconformity scores directly from the network
        learned_scores = scoring_model(cal_features).squeeze()  # [N]
        
        # Apply sophisticated tau calculation with stability measures
        n_cal = len(learned_scores)
        quantile_level = 1 - alpha
        
        # Add small random noise for stability (tie-breaking)
        noise = torch.randn_like(learned_scores) * 1e-6
        learned_scores_stable = learned_scores + noise
        
        # Calculate tau as quantile
        if n_cal > 0:
            # Use ceiling to be conservative (higher coverage)
            quantile_index = int(np.ceil((n_cal + 1) * quantile_level)) - 1
            quantile_index = max(0, min(quantile_index, n_cal - 1))
            
            sorted_scores, _ = torch.sort(learned_scores_stable)
            tau = sorted_scores[quantile_index]
        else:
            tau = torch.tensor(1.0).to(device)
            
        # Safety bounds for tau
        tau = torch.clamp(tau, min=0.1, max=10.0)
        
        if logger is not None:
            logger.info(f"Learned scores - min: {learned_scores.min().item():.6f}, "
                       f"max: {learned_scores.max().item():.6f}, "
                       f"mean: {learned_scores.mean().item():.6f}, "
                       f"std: {learned_scores.std().item():.6f}")
            logger.info(f"Calculated tau: {tau.item():.6f}")
        
    return tau


def determine_coverage_targets(scores: torch.Tensor, tau: torch.Tensor, 
                             gt_coords: torch.Tensor, pred_coords: torch.Tensor,
                             coverage_target: float = 0.9) -> torch.Tensor:
    """
    Determine which samples should be covered based on sophisticated criteria.
    
    Following classification framework: combines score-based and error-based criteria
    to create training targets for the separation loss.
    
    Args:
        scores: Current nonconformity scores [batch_size]
        tau: Current tau threshold
        gt_coords: Ground truth coordinates [batch_size, 4]
        pred_coords: Predicted coordinates [batch_size, 4]
        coverage_target: Target coverage level
        
    Returns:
        coverage_targets: Binary tensor [batch_size] indicating desired coverage
    """
    batch_size = scores.size(0)
    
    # Method 1: Score-based (samples with low scores should be covered)
    score_percentile = torch.quantile(scores, coverage_target)
    score_based_coverage = (scores <= score_percentile).float()
    
    # Method 2: Error-based (samples with low prediction errors should be covered)
    coord_errors = torch.abs(gt_coords - pred_coords).mean(dim=1)  # [batch_size]
    error_percentile = torch.quantile(coord_errors, coverage_target)
    error_based_coverage = (coord_errors <= error_percentile).float()
    
    # Method 3: Tau-based (current conformal prediction rule)
    tau_based_coverage = (scores <= tau).float()
    
    # Combine methods with weights (emphasize error-based for better learning)
    combined_coverage = (0.4 * error_based_coverage + 
                        0.3 * score_based_coverage + 
                        0.3 * tau_based_coverage)
    
    # Convert to binary targets (threshold at 0.5)
    coverage_targets = (combined_coverage >= 0.5).float()
    
    return coverage_targets


def train_epoch_sophisticated(scoring_model: ScoringMLP, optimizer: torch.optim.Optimizer,
                            train_data: dict, feature_extractor: FeatureExtractor, 
                            tau: torch.Tensor, lambda_width: float, margin_weight: float,
                            batch_size: int, device: torch.device, 
                            target_coverage: float = 0.9) -> dict:
    """
    Train for one epoch using sophisticated approach following classification framework.
    
    Args:
        scoring_model: Model to train
        optimizer: Optimizer
        train_data: Training data dictionary
        feature_extractor: Feature extractor
        tau: Current tau value
        lambda_width: Lambda for size penalty
        margin_weight: Weight for margin loss
        batch_size: Batch size
        device: Device
        target_coverage: Target coverage level
        
    Returns:
        metrics: Dictionary of training metrics
    """
    scoring_model.train()
    
    total_loss = 0.0
    total_coverage_loss = 0.0
    total_separation_loss = 0.0
    total_stability_loss = 0.0
    total_l2_loss = 0.0
    num_batches = 0
    
    # Create batches
    n_samples = train_data['features'].shape[0]
    indices = torch.randperm(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_indices = indices[i:batch_end]
        
        # Get batch data
        batch_features = train_data['features'][batch_indices].to(device)
        batch_gt = train_data['gt_coords'][batch_indices].to(device)
        batch_pred = train_data['pred_coords'][batch_indices].to(device)
        
        # Normalize features
        batch_features = feature_extractor.normalize_features(batch_features)
        
        # Determine coverage targets for this batch
        with torch.no_grad():
            temp_scores = scoring_model(batch_features).squeeze()
            coverage_targets = determine_coverage_targets(
                temp_scores, tau, batch_gt, batch_pred, target_coverage
            )
        
        # Forward pass with sophisticated loss computation
        optimizer.zero_grad()
        scores = scoring_model(batch_features, true_coverage_target=coverage_targets)
        
        # Main coverage loss
        coverage_criterion = CoverageLoss(
            target_coverage=target_coverage,
            lambda_width=lambda_width,
            margin_weight=margin_weight
        )
        coverage_loss = coverage_criterion(scores, tau, batch_gt, batch_pred)
        
        # Get sophisticated regularization losses from the model
        separation_loss = scoring_model.separation_loss
        stability_loss = scoring_model.stability_loss
        l2_loss = scoring_model.l2_reg
        
        # Total loss combination
        total_batch_loss = (coverage_loss + 
                           separation_loss + 
                           stability_loss + 
                           l2_loss)
        
        # Backward pass with gradient clipping
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(scoring_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate metrics
        total_loss += total_batch_loss.item()
        total_coverage_loss += coverage_loss.item()
        total_separation_loss += separation_loss.item() if hasattr(separation_loss, 'item') else 0.0
        total_stability_loss += stability_loss.item() if hasattr(stability_loss, 'item') else 0.0
        total_l2_loss += l2_loss.item() if hasattr(l2_loss, 'item') else 0.0
        num_batches += 1
    
    # Return comprehensive metrics
    return {
        'total_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'coverage_loss': total_coverage_loss / num_batches if num_batches > 0 else 0.0,
        'separation_loss': total_separation_loss / num_batches if num_batches > 0 else 0.0,
        'stability_loss': total_stability_loss / num_batches if num_batches > 0 else 0.0,
        'l2_loss': total_l2_loss / num_batches if num_batches > 0 else 0.0
    }


def validate_epoch_sophisticated(scoring_model: ScoringMLP, val_data: dict,
                               feature_extractor: FeatureExtractor, tau: torch.Tensor,
                               device: torch.device, target_coverage: float = 0.9) -> dict:
    """
    Validate model using sophisticated metrics following classification framework.
    
    Args:
        scoring_model: Model to validate
        val_data: Validation data dictionary
        feature_extractor: Feature extractor
        tau: Current tau value
        device: Device
        target_coverage: Target coverage level
        
    Returns:
        metrics: Dictionary of validation metrics
    """
    scoring_model.eval()
    
    with torch.no_grad():
        # Get validation features and coordinates
        val_features = feature_extractor.normalize_features(val_data['features'].to(device))
        val_gt = val_data['gt_coords'].to(device)
        val_pred = val_data['pred_coords'].to(device)
        
        # Get learned nonconformity scores
        scores = scoring_model(val_features).squeeze()  # [N]
        
        # Calculate coverage (percentage of samples with score <= tau)
        covered = (scores <= tau).float()
        actual_coverage = covered.mean().item()
        
        # Calculate average scores and their distribution
        avg_score = scores.mean().item()
        score_std = scores.std().item()
        score_min = scores.min().item()
        score_max = scores.max().item()
        
        # Calculate coordinate-wise prediction errors
        coord_errors = torch.abs(val_gt - val_pred)  # [N, 4]
        avg_coord_error = coord_errors.mean().item()
        
        # Calculate correlation between scores and prediction errors
        pred_errors = coord_errors.mean(dim=1)  # [N]
        if len(scores) > 1 and pred_errors.std() > 1e-6:
            # Compute correlation coefficient
            score_centered = scores - scores.mean()
            error_centered = pred_errors - pred_errors.mean()
            correlation = (score_centered * error_centered).mean() / (scores.std() * pred_errors.std())
        else:
            correlation = 0.0
        
        # Coverage efficiency (how close to target coverage)
        coverage_efficiency = 1.0 - abs(actual_coverage - target_coverage)
        
        # Calculate set size (for regression, this is related to prediction confidence)
        effective_set_size = covered.sum().item() / len(scores) if len(scores) > 0 else 0.0
        
    return {
        'coverage': actual_coverage,
        'avg_score': avg_score,
        'score_std': score_std,
        'score_min': score_min,
        'score_max': score_max,
        'avg_coord_error': avg_coord_error,
        'score_error_correlation': correlation.item() if hasattr(correlation, 'item') else correlation,
        'coverage_efficiency': coverage_efficiency,
        'effective_set_size': effective_set_size,
        'coverage_deviation': abs(actual_coverage - target_coverage)
    }


def plot_training_curves(metrics_history: dict, exp_dir: str):
    """Plot comprehensive training curves."""
    try:
        import matplotlib.pyplot as plt
        
        epochs = list(range(len(metrics_history['train_total_loss'])))
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(epochs, metrics_history['train_total_loss'], label='Train Total Loss', alpha=0.8)
        axes[0, 0].plot(epochs, metrics_history['train_coverage_loss'], label='Coverage Loss', alpha=0.8)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Coverage curves
        axes[0, 1].plot(epochs, metrics_history['val_coverage'], label='Validation Coverage', alpha=0.8)
        axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Target Coverage')
        axes[0, 1].set_title('Coverage Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Score statistics
        axes[0, 2].plot(epochs, metrics_history['val_avg_score'], label='Avg Score', alpha=0.8)
        axes[0, 2].plot(epochs, metrics_history['tau_values'], label='Tau', alpha=0.8)
        axes[0, 2].set_title('Score Statistics')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Regularization losses
        axes[1, 0].plot(epochs, metrics_history['train_separation_loss'], label='Separation Loss', alpha=0.8)
        axes[1, 0].plot(epochs, metrics_history['train_stability_loss'], label='Stability Loss', alpha=0.8)
        axes[1, 0].plot(epochs, metrics_history['train_l2_loss'], label='L2 Loss', alpha=0.8)
        axes[1, 0].set_title('Regularization Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Coverage efficiency
        axes[1, 1].plot(epochs, metrics_history['val_coverage_efficiency'], label='Coverage Efficiency', alpha=0.8)
        axes[1, 1].set_title('Coverage Efficiency')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Score-error correlation
        axes[1, 2].plot(epochs, metrics_history['val_score_error_correlation'], label='Score-Error Correlation', alpha=0.8)
        axes[1, 2].set_title('Score-Error Correlation')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'training_curves_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping plots")


def main():
    """Main training function with sophisticated approach."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    if args.exp_name is None:
        args.exp_name = f'sophisticated_scoring_{args.config_file}'
    
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting sophisticated training experiment: {args.exp_name}")
    logger.info(f"Arguments: {vars(args)}")

    # Load or prepare data
    if args.load_data is not None:
        logger.info(f"Loading preprocessed data from {args.load_data}")
        # Implementation for loading preprocessed data
        raise NotImplementedError("Data loading not implemented yet")
    else:
        # Load config and data
        cfg = io_file.load_yaml(args.config_file, args.config_path, to_yacs=True)
        data_name = cfg.DATASETS.DATASET.NAME
        
        # Register data with detectron2
        data_loader.d2_register_dataset(cfg, logger=logger)
        data_list = get_detection_dataset_dicts(
            data_name, filter_empty=cfg.DATASETS.DATASET.FILTER_EMPTY
        )
        
        # Set up predictions cache directory
        if args.predictions_cache_dir is None:
            cache_dir = os.path.join(exp_dir, 'predictions_cache')
        else:
            cache_dir = args.predictions_cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        img_list_path = os.path.join(cache_dir, f'{data_name}_img_list.json')
        ist_list_path = os.path.join(cache_dir, f'{data_name}_ist_list.json')
        
        # Load or collect predictions
        if args.load_predictions is not None:
            logger.info(f"Loading predictions from {args.load_predictions}")
            with open(os.path.join(args.load_predictions, f'{data_name}_img_list.json'), 'r') as f:
                img_list = json.load(f)
            with open(os.path.join(args.load_predictions, f'{data_name}_ist_list.json'), 'r') as f:
                ist_list = json.load(f)
        else:
            # Check if cached predictions exist
            if os.path.exists(img_list_path) and os.path.exists(ist_list_path):
                logger.info("Loading cached predictions...")
                with open(img_list_path, 'r') as f:
                    img_list = json.load(f)
                with open(ist_list_path, 'r') as f:
                    ist_list = json.load(f)
            else:
                logger.info("Collecting predictions...")
                img_list, ist_list = collect_predictions(
                    cfg=cfg,
                    data_list=data_list,
                    alpha=args.alpha,
                    label_set=args.label_set,
                    label_alpha=args.label_alpha,
                    risk_control=args.risk_control,
                    save_label_set=args.save_label_set,
                    logger=logger
                )
                
                # Save predictions if requested
                if args.save_predictions:
                    with open(img_list_path, 'w') as f:
                        json.dump(img_list, f)
                    with open(ist_list_path, 'w') as f:
                        json.dump(ist_list, f)
                    logger.info(f"Saved predictions to {cache_dir}")
        
        # Prepare training data (this function handles class frequency analysis internally)
        logger.info("Preparing training data...")
        all_gt_coords, all_pred_coords, all_pred_scores, selected_classes = prepare_training_data(
            ist_list, img_list, subset_size=args.subset_size
        )
        logger.info(f"Selected top classes: {selected_classes}")
        
        # Create feature extractor and extract features
        feature_extractor = FeatureExtractor()
        
        # For object detection, we need image dimensions - using default values for now
        # In a real implementation, you'd extract these from the actual images
        img_heights = torch.full((len(all_gt_coords),), 480.0)  # Default height
        img_widths = torch.full((len(all_gt_coords),), 640.0)   # Default width
        
        all_features = feature_extractor.extract_features(
            all_pred_coords, all_pred_scores, img_heights, img_widths
        )
        
        # Fit normalizer on all features
        feature_extractor.fit_normalizer(all_features)
        
        logger.info(f"Extracted features shape: {all_features.shape}")
        logger.info(f"Feature names: {get_feature_names()}")
        
        # Split data
        train_data, cal_data, val_data = split_data(
            features=all_features,
            gt_coords=all_gt_coords,
            pred_coords=all_pred_coords,
            train_frac=args.train_frac,
            cal_frac=args.cal_frac,
            val_frac=args.val_frac
        )
        
        logger.info(f"Train data: {train_data['features'].shape[0]} samples")
        logger.info(f"Cal data: {cal_data['features'].shape[0]} samples")
        logger.info(f"Val data: {val_data['features'].shape[0]} samples")
        
        # Validate feature dimensions
        expected_dim = 13
        actual_dim = train_data['features'].shape[1]
        if actual_dim != expected_dim:
            logger.error(f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}")
            raise ValueError(f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}")
        
        # Save data if requested
        if args.save_data:
            data_path = os.path.join(exp_dir, 'training_data.pt')
            
            # Check if data already exists and validate it
            if os.path.exists(data_path):
                try:
                    existing_data = torch.load(data_path)
                    existing_features = existing_data.get('train_data', {}).get('features')
                    if existing_features is not None and existing_features.shape[1] != 13:
                        logger.warning(f"Existing cached data has wrong feature dimensions ({existing_features.shape[1]}), removing it...")
                        os.remove(data_path)
                except Exception as e:
                    logger.warning(f"Could not load existing data file, will overwrite: {e}")
                    if os.path.exists(data_path):
                        os.remove(data_path)
            
            torch.save({
                'train_data': train_data,
                'cal_data': cal_data,
                'val_data': val_data,
                'feature_extractor': feature_extractor,
                'selected_classes': selected_classes
            }, data_path)
            logger.info(f"Saved training data to {data_path}")
    
    # Initialize sophisticated model
    logger.info("Initializing sophisticated scoring model...")
    model_config = {
        'input_dim': args.input_dim,
        'hidden_dims': args.hidden_dims,
        'output_dim': 1,
        'dropout_rate': args.dropout_rate,
        'config': {
            'l2_lambda': 0.001,
            'stability_factor': 0.1,
            'separation_factor': 1.0
        }
    }
    
    scoring_model = ScoringMLP(**model_config).to(device)
    optimizer = optim.AdamW(
        scoring_model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # Initialize sophisticated scheduler
    lambda_scheduler = AdaptiveLambdaScheduler(
        args.initial_lambda, args.final_lambda, 
        args.warmup_epochs, args.ramp_epochs,
        schedule_type=args.schedule_type
    )
    
    logger.info(f"Model info: {scoring_model.get_model_info()}")
    
    # Initialize metrics tracking
    metrics_history = {
        'train_total_loss': [],
        'train_coverage_loss': [],
        'train_separation_loss': [],
        'train_stability_loss': [],
        'train_l2_loss': [],
        'val_coverage': [],
        'val_avg_score': [],
        'val_coverage_efficiency': [],
        'val_score_error_correlation': [],
        'tau_values': []
    }
    
    # Training loop with sophisticated approach
    logger.info("Starting sophisticated training...")
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Update tau periodically using sophisticated calculation
        if epoch % args.tau_update_freq == 0:
            tau = calculate_sophisticated_tau(
                scoring_model, cal_data, feature_extractor, 
                alpha=args.alpha, device=device, logger=logger
            )
            scoring_model.set_tau(tau.item())
        
        logger.info(f"Current tau: {tau.item():.6f}")
        
        # Get current lambda for this epoch
        current_lambda = lambda_scheduler.get_lambda(epoch)
        logger.info(f"Current lambda: {current_lambda:.6f}")
        
        # Train epoch with sophisticated approach
        train_metrics = train_epoch_sophisticated(
            scoring_model, optimizer, train_data, feature_extractor,
            tau, current_lambda, args.margin_weight, args.batch_size, device, args.target_coverage
        )
        
        # Validate with sophisticated metrics
        val_metrics = validate_epoch_sophisticated(
            scoring_model, val_data, feature_extractor, tau, device, args.target_coverage
        )
        
        # Log comprehensive metrics
        logger.info(f"Train - Total: {train_metrics['total_loss']:.4f}, "
                   f"Coverage: {train_metrics['coverage_loss']:.4f}, "
                   f"Separation: {train_metrics['separation_loss']:.4f}")
        logger.info(f"Val - Coverage: {val_metrics['coverage']:.4f}, "
                   f"Avg Score: {val_metrics['avg_score']:.4f}, "
                   f"Efficiency: {val_metrics['coverage_efficiency']:.4f}")
        
        # Update metrics history
        metrics_history['train_total_loss'].append(train_metrics['total_loss'])
        metrics_history['train_coverage_loss'].append(train_metrics['coverage_loss'])
        metrics_history['train_separation_loss'].append(train_metrics['separation_loss'])
        metrics_history['train_stability_loss'].append(train_metrics['stability_loss'])
        metrics_history['train_l2_loss'].append(train_metrics['l2_loss'])
        metrics_history['val_coverage'].append(val_metrics['coverage'])
        metrics_history['val_avg_score'].append(val_metrics['avg_score'])
        metrics_history['val_coverage_efficiency'].append(val_metrics['coverage_efficiency'])
        metrics_history['val_score_error_correlation'].append(val_metrics['score_error_correlation'])
        metrics_history['tau_values'].append(tau.item())
        
        # Early stopping based on coverage efficiency
        val_loss = val_metrics['coverage_deviation']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            save_model(
                scoring_model, optimizer, epoch, val_loss, model_config,
                os.path.join(exp_dir, 'best_sophisticated_model.pt'),
                feature_extractor.feature_stats
            )
            logger.info(f"New best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            save_model(
                scoring_model, optimizer, epoch, val_loss, model_config,
                os.path.join(exp_dir, f'checkpoint_epoch_{epoch + 1}.pt'),
                feature_extractor.feature_stats
            )
    
    # Save comprehensive results
    final_metrics = {
        'metrics_history': metrics_history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_tau': tau.item(),
        'model_config': model_config,
        'training_args': vars(args)
    }
    
    with open(os.path.join(exp_dir, 'sophisticated_training_results.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Plot comprehensive training curves
    plot_training_curves(metrics_history, exp_dir)
    
    # Save feature extractor
    feature_extractor.save_stats(os.path.join(exp_dir, 'feature_stats.pt'))
    
    logger.info(f"Sophisticated training completed!")
    logger.info(f"Best epoch: {best_epoch + 1}")
    logger.info(f"Best validation coverage deviation: {best_val_loss:.4f}")
    logger.info(f"Final coverage: {metrics_history['val_coverage'][-1]:.4f}")
    logger.info(f"Final coverage efficiency: {metrics_history['val_coverage_efficiency'][-1]:.4f}")
    
    return exp_dir


if __name__ == "__main__":
    main() 