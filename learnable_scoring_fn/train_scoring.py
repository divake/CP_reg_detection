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

# Add project paths
sys.path.insert(0, "/ssd_4TB/divake/conformal-od")
sys.path.insert(0, "/ssd_4TB/divake/conformal-od/detectron2")

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts

# Local imports
from .model import ScoringMLP, CoverageLoss, AdaptiveLambdaScheduler, save_model, load_model
from .feature_utils import FeatureExtractor, get_feature_names
from .data_utils import prepare_training_data, split_data, COCOClassMapper
from data import data_loader
from model import model_loader
from control.std_conformal import StdConformal
from util import util, io_file


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
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Coverage loss parameters
    parser.add_argument('--target_coverage', type=float, default=0.9, help='Target coverage level')
    parser.add_argument('--initial_lambda', type=float, default=0.01, help='Initial lambda for coverage loss')
    parser.add_argument('--final_lambda', type=float, default=0.1, help='Final lambda for coverage loss')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--ramp_epochs', type=int, default=20, help='Number of ramp epochs for lambda schedule')
    
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


def calculate_tau_per_class(scoring_model: ScoringMLP, cal_data: dict, 
                           feature_extractor: FeatureExtractor,
                           alpha: float = 0.1, device: torch.device = None, logger=None) -> torch.Tensor:
    """
    Calculate tau (quantile threshold) per class using calibration data.
    
    This follows the classification framework pattern where tau is calculated
    per epoch using the calibration set.
    
    Args:
        scoring_model: Trained scoring model
        cal_data: Calibration data dictionary
        feature_extractor: Feature extractor
        alpha: Miscoverage level (1-coverage)
        device: Device to use
        
    Returns:
        tau: Quantile threshold tensor [4] for each coordinate
    """
    scoring_model.eval()
    
    with torch.no_grad():
        # Extract features and get scores
        cal_features = feature_extractor.normalize_features(cal_data['features'].to(device))
        
        # Debug: Check dimensions before passing to model
        if logger is not None:
            logger.info(f"Cal features shape before normalization: {cal_data['features'].shape}")
            logger.info(f"Cal features shape after normalization: {cal_features.shape}")
        
        # Safety check for dimensions before model forward pass
        expected_input_dim = scoring_model.input_dim
        actual_input_dim = cal_features.shape[1]
        if actual_input_dim != expected_input_dim:
            error_msg = f"Feature dimension mismatch: model expects {expected_input_dim} but got {actual_input_dim}. "
            error_msg += "This suggests corrupted cached data or incorrect feature extraction. "
            error_msg += "Try deleting cached data and re-running."
            if logger is not None:
                logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        cal_scores = scoring_model(cal_features).squeeze()  # [N]
        
        # Debug: Check scoring model outputs
        if logger is not None:
            logger.info(f"Scoring model outputs - min: {cal_scores.min().item():.6f}, max: {cal_scores.max().item():.6f}, mean: {cal_scores.mean().item():.6f}")
        
        # Calculate absolute errors between GT and predictions
        cal_abs_errors = torch.abs(cal_data['gt_coords'] - cal_data['pred_coords']).to(device)  # [N, 4]
        
        # Debug: Check absolute errors
        if logger is not None:
            logger.info(f"Cal absolute errors - min: {cal_abs_errors.min().item():.6f}, max: {cal_abs_errors.max().item():.6f}, mean: {cal_abs_errors.mean().item():.6f}")
        
        # Calculate nonconformity scores: error / learned_score for each coordinate
        # Broadcast cal_scores to match coordinate dimensions
        cal_scores_broadcast = cal_scores.unsqueeze(1).expand(-1, 4)  # [N, 4]
        
        # Add larger epsilon to prevent division by very small numbers
        epsilon = 1e-3  # Increased from 1e-8
        nonconf_scores = cal_abs_errors / (cal_scores_broadcast + epsilon)  # [N, 4]
        
        # Debug: Check nonconformity scores
        if logger is not None:
            logger.info(f"Nonconformity scores - min: {nonconf_scores.min().item():.6f}, max: {nonconf_scores.max().item():.6f}, mean: {nonconf_scores.mean().item():.6f}")
        
        # Calculate quantile (tau) for each coordinate
        quantile_level = 1 - alpha
        taus = torch.quantile(nonconf_scores, quantile_level, dim=0)  # [4]
        
    return taus


def train_epoch(scoring_model: ScoringMLP, optimizer: torch.optim.Optimizer,
               criterion: CoverageLoss, train_data: dict, 
               feature_extractor: FeatureExtractor, tau: torch.Tensor,
               batch_size: int, device: torch.device) -> float:
    """
    Train for one epoch with fixed tau.
    
    Args:
        scoring_model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        train_data: Training data
        feature_extractor: Feature extractor
        tau: Fixed tau values for this epoch [4]
        batch_size: Batch size
        device: Device
        
    Returns:
        average_loss: Average loss for the epoch
    """
    scoring_model.train()
    total_loss = 0.0
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
        
        # Forward pass
        optimizer.zero_grad()
        learned_scores = scoring_model(batch_features).squeeze()  # [batch_size]
        
        # Add regularization to prevent very small scores
        score_penalty = torch.mean(torch.relu(1.0 - learned_scores))  # Penalty for scores < 1.0
        
        # Create prediction intervals using learned scores and fixed tau
        learned_scores_broadcast = learned_scores.unsqueeze(1).expand(-1, 4)  # [batch_size, 4]
        tau_broadcast = tau.unsqueeze(0).expand(batch_features.shape[0], -1)  # [batch_size, 4]
        
        interval_widths = learned_scores_broadcast * tau_broadcast  # [batch_size, 4]
        pred_lower = batch_pred - interval_widths
        pred_upper = batch_pred + interval_widths
        
        # Calculate loss
        coverage_loss = criterion(pred_lower, pred_upper, batch_gt)
        loss = coverage_loss + 0.1 * score_penalty  # Add regularization
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scoring_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(scoring_model: ScoringMLP, val_data: dict,
                  feature_extractor: FeatureExtractor, tau: torch.Tensor,
                  device: torch.device) -> tuple:
    """
    Validate model performance with current tau.
    
    Args:
        scoring_model: Model to validate
        val_data: Validation data
        feature_extractor: Feature extractor
        tau: Current tau values [4]
        device: Device
        
    Returns:
        coverage: Average coverage across coordinates
        avg_width: Average interval width
        val_loss: Validation loss
    """
    scoring_model.eval()
    
    with torch.no_grad():
        # Get validation features and predictions
        val_features = feature_extractor.normalize_features(val_data['features'].to(device))
        val_gt = val_data['gt_coords'].to(device)
        val_pred = val_data['pred_coords'].to(device)
        
        # Get learned scores
        learned_scores = scoring_model(val_features).squeeze()  # [N]
        
        # Create prediction intervals
        learned_scores_broadcast = learned_scores.unsqueeze(1).expand(-1, 4)  # [N, 4]
        tau_broadcast = tau.unsqueeze(0).expand(val_features.shape[0], -1)  # [N, 4]
        
        interval_widths = learned_scores_broadcast * tau_broadcast
        pred_lower = val_pred - interval_widths
        pred_upper = val_pred + interval_widths
        
        # Calculate coverage
        covered = (val_gt >= pred_lower) & (val_gt <= pred_upper)
        coverage = covered.float().mean().item()
        
        # Calculate average width (relative to coordinate magnitudes)
        avg_width = interval_widths.mean().item()
        coord_magnitudes = torch.abs(val_pred).mean().item()
        relative_width = avg_width / (coord_magnitudes + 1e-6)
        
        # Calculate validation loss
        criterion = CoverageLoss(target_coverage=0.9, lambda_width=0.1)
        val_loss = criterion(pred_lower, pred_upper, val_gt).item()
    
    return coverage, relative_width, val_loss


def plot_training_curves(metrics: dict, output_dir: str):
    """Plot and save training curves."""
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['train_losses'], label='Training Loss')
    plt.plot(epochs, metrics['val_losses'], label='Validation Loss')
    plt.axvline(x=metrics['best_epoch'] + 1, color='r', linestyle='--', label=f'Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Coverage curve
    plt.subplot(1, 3, 2)
    coverages = [m[0] for m in metrics['coverage_metrics']]
    plt.plot(epochs, coverages, label='Validation Coverage', color='green')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target Coverage (90%)')
    plt.axvline(x=metrics['best_epoch'] + 1, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Coverage')
    plt.title('Validation Coverage')
    plt.legend()
    plt.grid(True)
    
    # Width curve  
    plt.subplot(1, 3, 3)
    widths = [m[1] for m in metrics['coverage_metrics']]
    plt.plot(epochs, widths, label='Relative Width', color='orange')
    plt.axvline(x=metrics['best_epoch'] + 1, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Relative Width')
    plt.title('Prediction Interval Width')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up experiment directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"learnable_scoring_{timestamp}"
    
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(output=exp_dir)
    logger.info(f"Starting experiment: {args.exp_name}")
    logger.info(f"Output directory: {exp_dir}")
    
    # Save experiment configuration
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device and seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    util.set_seed(args.seed, logger=logger)
    logger.info(f"Using device: {device}")
    
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
        if args.load_predictions and os.path.exists(img_list_path) and os.path.exists(ist_list_path):
            logger.info(f"Loading cached predictions from {cache_dir}")
            img_list = io_file.load_json(f'{data_name}_img_list', cache_dir)
            ist_list = io_file.load_json(f'{data_name}_ist_list', cache_dir)
            
            # We still need to build the model for feature extraction
            cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
            model_loader.d2_load_model(cfg_model, model, logger=logger)
            
        else:
            logger.info("Collecting predictions (this may take 6-10 minutes)...")
            
            # Build model config before creating dataloader
            cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
            model_loader.d2_load_model(cfg_model, model, logger=logger)
            
            dataloader = data_loader.d2_load_dataset_from_dict(
                data_list, cfg, cfg_model, logger=logger
            )
            metadata = MetadataCatalog.get(data_name).as_dict()
            nr_class = len(metadata["thing_classes"])
            nr_img = len(data_list)
            
            # Initialize and run data collection
            controller = StdConformal(cfg, args, nr_class, exp_dir, log=None, logger=logger)
            controller.set_collector(nr_class, nr_img)
            
            img_list, ist_list = controller.collect_predictions(model, dataloader, verbose=False)
            
            # Save predictions if requested
            if args.save_predictions:
                logger.info(f"Saving predictions cache to {cache_dir}")
                io_file.save_json(img_list, f'{data_name}_img_list', cache_dir)
                io_file.save_json(ist_list, f'{data_name}_ist_list', cache_dir)
                logger.info("Predictions cached for future use")
        
        # Prepare training data
        logger.info("Preparing training data...")
        gt_coords, pred_coords, pred_scores, selected_classes = prepare_training_data(
            ist_list, img_list, args.subset_size
        )
        
        # Extract features
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(pred_coords, pred_scores)
        
        # Debug: Check feature dimensions
        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Expected feature dimensions: 13")
        
        # Safety check: Ensure features have correct dimensions
        if features.shape[1] != 13:
            error_msg = f"Feature extraction produced {features.shape[1]} dimensions, expected 13. "
            error_msg += "This suggests an issue with the FeatureExtractor. "
            error_msg += f"Feature shape: {features.shape}, pred_coords shape: {pred_coords.shape}, pred_scores shape: {pred_scores.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Fit normalizer on all features
        feature_extractor.fit_normalizer(features)
        
        # Split data
        train_data, cal_data, val_data = split_data(
            features, gt_coords, pred_coords,
            args.train_frac, args.cal_frac, args.val_frac, args.seed
        )
        
        logger.info(f"Training data: {train_data['features'].shape[0]} samples")
        logger.info(f"Calibration data: {cal_data['features'].shape[0]} samples")
        logger.info(f"Validation data: {val_data['features'].shape[0]} samples")
        
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
    
    # Initialize model
    logger.info("Initializing model...")
    model_config = {
        'input_dim': args.input_dim,
        'hidden_dims': args.hidden_dims,
        'output_dim': 1,
        'dropout_rate': args.dropout_rate
    }
    
    scoring_model = ScoringMLP(**model_config).to(device)
    optimizer = optim.Adam(scoring_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Initialize loss and scheduler
    lambda_scheduler = AdaptiveLambdaScheduler(
        args.initial_lambda, args.final_lambda, args.warmup_epochs, args.ramp_epochs
    )
    
    logger.info(f"Model info: {scoring_model.get_model_info()}")
    
    # Training loop
    logger.info("Starting training...")
    
    train_losses = []
    val_losses = []
    coverage_metrics = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 1. Calibration phase: Calculate tau using calibration set
        tau = calculate_tau_per_class(scoring_model, cal_data, feature_extractor, alpha=0.1, device=device, logger=logger)
        logger.info(f"Calculated tau: {tau.cpu().numpy()}")
        
        # Safety check: stop training if tau values become too large
        if torch.any(tau > 1000):
            logger.warning(f"Tau values too large: {tau.cpu().numpy()}, stopping training early")
            break
        
        # 2. Training phase: Train with fixed tau
        current_lambda = lambda_scheduler.get_lambda(epoch)
        criterion = CoverageLoss(target_coverage=args.target_coverage, lambda_width=current_lambda)
        
        train_loss = train_epoch(
            scoring_model, optimizer, criterion, train_data, 
            feature_extractor, tau, args.batch_size, device
        )
        train_losses.append(train_loss)
        
        # 3. Validation phase: Evaluate with learned scores and current tau
        coverage, width, val_loss = validate_epoch(
            scoring_model, val_data, feature_extractor, tau, device
        )
        val_losses.append(val_loss)
        coverage_metrics.append((coverage, width))
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Lambda: {current_lambda:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(
                scoring_model, optimizer, epoch, val_loss, model_config,
                os.path.join(exp_dir, 'best_model.pt'),
                feature_extractor.feature_stats
            )
            logger.info(f"New best model saved at epoch {epoch + 1}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_model(
                scoring_model, optimizer, epoch, val_loss, model_config,
                os.path.join(exp_dir, f'checkpoint_epoch_{epoch + 1}.pt'),
                feature_extractor.feature_stats
            )
    
    # Save final results
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'coverage_metrics': coverage_metrics,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_tau': tau.cpu().numpy().tolist()
    }
    
    with open(os.path.join(exp_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training curves
    plot_training_curves(metrics, exp_dir)
    
    # Save feature extractor
    feature_extractor.save_stats(os.path.join(exp_dir, 'feature_stats.pt'))
    
    logger.info(f"Training completed. Best epoch: {best_epoch + 1}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final coverage: {coverage_metrics[-1][0]:.4f}")
    logger.info(f"Final width: {coverage_metrics[-1][1]:.4f}")
    
    return exp_dir


if __name__ == "__main__":
    main() 