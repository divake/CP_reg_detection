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
    
    # Data arguments
    parser.add_argument("--config_file", type=str, default="cfg_std_rank", 
                       help="Config file name")
    parser.add_argument("--config_path", type=str, default="/ssd_4TB/divake/conformal-od/config/coco_val",
                       help="Path to config directory")
    parser.add_argument("--subset_size", type=int, default=50000,
                       help="Size of training subset")
    
    # Model arguments  
    parser.add_argument("--input_dim", type=int, default=13,
                       help="Input feature dimension")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64, 32],
                       help="Hidden layer dimensions")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                       help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    
    # Loss function arguments
    parser.add_argument("--target_coverage", type=float, default=0.9,
                       help="Target coverage level")
    parser.add_argument("--initial_lambda", type=float, default=0.01,
                       help="Initial lambda for width penalty")
    parser.add_argument("--final_lambda", type=float, default=0.1,
                       help="Final lambda for width penalty")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                       help="Epochs for lambda warmup")
    parser.add_argument("--ramp_epochs", type=int, default=30,
                       help="Epochs to ramp lambda")
    
    # Data split arguments
    parser.add_argument("--train_frac", type=float, default=0.5,
                       help="Training data fraction")
    parser.add_argument("--cal_frac", type=float, default=0.3,
                       help="Calibration data fraction")
    parser.add_argument("--val_frac", type=float, default=0.2,
                       help="Validation data fraction")
    
    # I/O arguments
    parser.add_argument("--output_dir", type=str, default="/ssd_4TB/divake/conformal-od/learnable_scoring_fn/experiments",
                       help="Output directory for experiments")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Experiment name (auto-generated if None)")
    parser.add_argument("--load_data", type=str, default=None,
                       help="Path to load preprocessed data")
    parser.add_argument("--save_data", action="store_true",
                       help="Save preprocessed data")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    
    return parser


def calculate_tau_per_class(scoring_model: ScoringMLP, cal_data: dict, 
                           feature_extractor: FeatureExtractor,
                           alpha: float = 0.1, device: torch.device = None) -> torch.Tensor:
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
        cal_scores = scoring_model(cal_features).squeeze()  # [N]
        
        # Calculate absolute errors between GT and predictions
        cal_abs_errors = torch.abs(cal_data['gt_coords'] - cal_data['pred_coords']).to(device)  # [N, 4]
        
        # Calculate nonconformity scores: error / learned_score for each coordinate
        # Broadcast cal_scores to match coordinate dimensions
        cal_scores_broadcast = cal_scores.unsqueeze(1).expand(-1, 4)  # [N, 4]
        nonconf_scores = cal_abs_errors / (cal_scores_broadcast + 1e-8)  # [N, 4]
        
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
        
        # Create prediction intervals using learned scores and fixed tau
        learned_scores_broadcast = learned_scores.unsqueeze(1).expand(-1, 4)  # [batch_size, 4]
        tau_broadcast = tau.unsqueeze(0).expand(batch_features.shape[0], -1)  # [batch_size, 4]
        
        interval_widths = learned_scores_broadcast * tau_broadcast  # [batch_size, 4]
        pred_lower = batch_pred - interval_widths
        pred_upper = batch_pred + interval_widths
        
        # Calculate loss
        loss = criterion(pred_lower, pred_upper, batch_gt)
        
        # Backward pass
        loss.backward()
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
        dataloader = data_loader.d2_load_dataset_from_dict(
            data_list, cfg, None, logger=logger
        )
        metadata = MetadataCatalog.get(data_name).as_dict()
        nr_class = len(metadata["thing_classes"])
        nr_img = len(data_list)
        
        # Initialize and run data collection
        logger.info("Collecting predictions for training data...")
        controller = StdConformal(cfg, args, nr_class, exp_dir, log=None, logger=logger)
        controller.set_collector(nr_class, nr_img)
        
        # Load model and collect predictions
        cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
        model_loader.d2_load_model(cfg_model, model, logger=logger)
        
        img_list, ist_list = controller.collect_predictions(model, dataloader, verbose=False)
        
        # Prepare training data
        logger.info("Preparing training data...")
        gt_coords, pred_coords, pred_scores, selected_classes = prepare_training_data(
            ist_list, img_list, args.subset_size
        )
        
        # Extract features
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(pred_coords, pred_scores)
        
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
        tau = calculate_tau_per_class(scoring_model, cal_data, feature_extractor, alpha=0.1, device=device)
        logger.info(f"Calculated tau: {tau.cpu().numpy()}")
        
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