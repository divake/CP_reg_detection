#!/usr/bin/env python3
"""
Main training script for regression-based learnable scoring function.

FIXED IMPLEMENTATION with correct coverage definition:
- Coverage = P(gt ∈ [pred - width*tau, pred + width*tau])
- Fixed tau = 1.0 (model learns appropriate widths)
- Proper efficiency and calibration losses

Data splits:
- COCO train set for training the scoring function
- COCO val set split into calibration and test sets
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import json
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Add project paths
project_root = "/ssd_4TB/divake/conformal-od"
detectron2_path = "/ssd_4TB/divake/conformal-od/detectron2"

sys.path.insert(0, detectron2_path)
sys.path.insert(0, project_root)

os.environ['DETECTRON2_DATASETS'] = '/ssd_4TB/divake/conformal-od/data'

# Import components
from learnable_scoring_fn.model import (
    RegressionScoringFunction, RegressionCoverageLoss,
    calculate_tau_regression, UncertaintyFeatureExtractor,
    save_regression_model
)
from learnable_scoring_fn.feature_utils import FeatureExtractor, get_feature_names

# Import project components
from util import io_file
from util.util import set_seed, set_device
from data import data_loader
from model import model_loader
from detectron2.data import get_detection_dataset_dicts, MetadataCatalog
from control.std_conformal import StdConformal
from calibration.random_split import random_split


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_existing_predictions(dataset_name, output_dir="/ssd_4TB/divake/conformal-od/output"):
    """Load existing predictions from standard conformal runs."""
    base_dir = Path(output_dir) / dataset_name
    
    # Look for existing prediction files
    possible_dirs = [
        "std_conf_x101fpn_std_rank_class",
        "std_conf_x101fpn_std_bonf_class",
        "std_conf_x101fpn",
    ]
    
    for dir_name in possible_dirs:
        pred_dir = base_dir / dir_name
        img_file = pred_dir / f"{dir_name}_img_list.json"
        ist_file = pred_dir / f"{dir_name}_ist_list.json"
        
        if img_file.exists() and ist_file.exists():
            print(f"Found existing predictions in {pred_dir}")
            with open(img_file, 'r') as f:
                img_list = json.load(f)
            with open(ist_file, 'r') as f:
                ist_list = json.load(f)
            return img_list, ist_list
    
    return None, None


def collect_predictions_for_dataset(cfg_file, cfg_path, dataset_type, cache_dir=None, logger=None):
    """
    Collect predictions for a specific dataset (train or val).
    
    Args:
        cfg_file: Base config file name
        cfg_path: Path to config directory
        dataset_type: 'train' or 'val'
        cache_dir: Directory to cache predictions
        logger: Logger instance
    """
    if logger:
        logger.info(f"Collecting predictions for {dataset_type} dataset...")
    
    # Check cache first
    if cache_dir:
        cache_file = Path(cache_dir) / f"predictions_{dataset_type}.pkl"
        if cache_file.exists():
            if logger:
                logger.info(f"Loading cached {dataset_type} predictions...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Modify config path based on dataset type
    if dataset_type == 'train':
        cfg_path = cfg_path.replace('coco_val', 'coco_train')
        # For training, we might need to use a different config
        # Check if train config exists
        train_cfg_file = Path(cfg_path) / f"{cfg_file}.yaml"
        if not train_cfg_file.exists():
            # Use QR training config as base
            cfg_file = 'cfg_qr_train'
    
    # Load configuration
    cfg = io_file.load_yaml(cfg_file, cfg_path, to_yacs=True)
    
    # Override dataset name if needed
    if dataset_type == 'train':
        cfg.DATASETS.DATASET.NAME = 'coco_train'
        cfg.DATASETS.DATASET.IMG_DIR = 'coco/train2017'
        cfg.DATASETS.DATASET.ANN_FILE = 'coco/annotations/instances_train2017.json'
        # Use local checkpoint for train dataset to avoid model zoo error
        cfg.MODEL.LOCAL_CHECKPOINT = True
        cfg.MODEL.CHECKPOINT_PATH = 'checkpoints/x101fpn_train_qr_5k_postprocess.pth'
        # Also use a local config file path
        cfg.MODEL.FILE = '/ssd_4TB/divake/conformal-od/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        # Use StandardROIHeads instead of QuantileROIHead
        cfg.MODEL.CONFIG.MODEL.ROI_HEADS.NAME = 'StandardROIHeads'
    
    data_name = cfg.DATASETS.DATASET.NAME
    
    # Check if predictions already exist
    if dataset_type == 'val':
        img_list, ist_list = load_existing_predictions(data_name)
        if img_list is not None and ist_list is not None:
            if logger:
                logger.info(f"Using existing predictions for {data_name}")
            
            # Cache the predictions
            if cache_dir:
                cache_file = Path(cache_dir) / f"predictions_{dataset_type}.pkl"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump((img_list, ist_list), f)
            
            return img_list, ist_list
    
    # Otherwise, collect new predictions
    set_seed(cfg.PROJECT.SEED, logger=logger)
    cfg, device = set_device(cfg, 'cuda', logger=logger)
    
    # Register dataset
    data_loader.d2_register_dataset(cfg, logger=logger)
    
    # Build and load model
    cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
    model_loader.d2_load_model(cfg_model, model, logger=logger)
    model.eval()
    
    # Load dataset
    data_list = get_detection_dataset_dicts(
        data_name, 
        filter_empty=cfg.DATASETS.DATASET.FILTER_EMPTY
    )
    
    # Limit dataset size for faster iteration
    if dataset_type == 'train':
        data_list = data_list[:2000]  # Use first 2000 training images
    else:
        data_list = data_list[:1000]  # Use first 1000 val images
    
    dataloader = data_loader.d2_load_dataset_from_dict(
        data_list, cfg, cfg_model, logger=logger
    )
    
    metadata = MetadataCatalog.get(data_name).as_dict()
    nr_class = len(metadata["thing_classes"])
    
    # Create args for StdConformal
    class Args:
        def __init__(self):
            self.alpha = 0.1
            self.label_set = 'top_singleton'
            self.label_alpha = 0.1
            self.risk_control = False
            self.save_label_set = False
    
    args = Args()
    
    # Use StdConformal to collect predictions
    controller = StdConformal(
        cfg=cfg,
        args=args,
        nr_class=nr_class,
        filedir='.',
        log=None,
        logger=logger
    )
    
    controller.set_collector(nr_class, len(data_list))
    img_list, ist_list = controller.collect_predictions(model, dataloader)
    
    # Cache predictions
    if cache_dir:
        cache_file = Path(cache_dir) / f"predictions_{dataset_type}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((img_list, ist_list), f)
        if logger:
            logger.info(f"Cached {dataset_type} predictions to {cache_file}")
    
    return img_list, ist_list


def prepare_regression_data(img_list, ist_list, logger):
    """Prepare data for regression training from collected predictions."""
    logger.info("Preparing regression training data...")
    
    feature_extractor = FeatureExtractor()
    uncertainty_extractor = UncertaintyFeatureExtractor()
    
    all_features = []
    all_pred_coords = []
    all_gt_coords = []
    all_errors = []
    all_confidence = []
    all_img_ids = []
    
    # Process each class
    for class_id in range(len(ist_list)):
        if 'pred_x0' not in ist_list[class_id]:
            continue
            
        class_data = ist_list[class_id]
        n_preds = len(class_data['pred_x0'])
        
        if n_preds == 0:
            continue
        
        # Convert to tensors
        pred_x0 = torch.tensor(class_data['pred_x0'], dtype=torch.float32)
        pred_y0 = torch.tensor(class_data['pred_y0'], dtype=torch.float32)
        pred_x1 = torch.tensor(class_data['pred_x1'], dtype=torch.float32)
        pred_y1 = torch.tensor(class_data['pred_y1'], dtype=torch.float32)
        pred_scores = torch.tensor(class_data['pred_score'], dtype=torch.float32)
        
        gt_x0 = torch.tensor(class_data['gt_x0'], dtype=torch.float32)
        gt_y0 = torch.tensor(class_data['gt_y0'], dtype=torch.float32)
        gt_x1 = torch.tensor(class_data['gt_x1'], dtype=torch.float32)
        gt_y1 = torch.tensor(class_data['gt_y1'], dtype=torch.float32)
        
        img_ids = torch.tensor(class_data['img_id'], dtype=torch.int64)
        
        # Stack coordinates
        pred_coords = torch.stack([pred_x0, pred_y0, pred_x1, pred_y1], dim=1)
        gt_coords = torch.stack([gt_x0, gt_y0, gt_x1, gt_y1], dim=1)
        
        # Extract features for each detection
        for i in range(n_preds):
            # Get features
            features = feature_extractor.extract_features(
                pred_coords[i:i+1],
                pred_scores[i:i+1]
            ).squeeze(0)
            
            # Calculate error
            error = torch.abs(gt_coords[i] - pred_coords[i])
            
            all_features.append(features)
            all_pred_coords.append(pred_coords[i])
            all_gt_coords.append(gt_coords[i])
            all_errors.append(error)
            all_confidence.append(pred_scores[i])
            all_img_ids.append(img_ids[i])
    
    if not all_features:
        raise ValueError("No valid prediction-GT pairs found!")
    
    # Stack all
    features = torch.stack(all_features)
    pred_coords = torch.stack(all_pred_coords)
    gt_coords = torch.stack(all_gt_coords)
    errors = torch.stack(all_errors)
    confidence = torch.stack(all_confidence)
    img_ids = torch.stack(all_img_ids)
    
    # Fit error distribution
    uncertainty_extractor.fit_error_distribution(errors)
    
    # Extract uncertainty features
    uncertainty_features = uncertainty_extractor.extract_uncertainty_features(
        pred_coords, confidence
    )
    
    # Combine features
    combined_features = torch.cat([features, uncertainty_features], dim=1)
    
    # Normalize features
    feature_extractor.fit_normalizer(combined_features)
    combined_features = feature_extractor.normalize_features(combined_features)
    
    logger.info(f"Prepared {len(combined_features)} samples")
    logger.info(f"Feature dimension: {combined_features.shape[1]}")
    logger.info(f"Average error: {errors.mean():.3f}")
    
    return combined_features, pred_coords, gt_coords, errors, img_ids, feature_extractor, uncertainty_extractor


def split_val_data(features, pred_coords, gt_coords, errors, img_ids, calib_fraction=0.5):
    """
    Split validation data into calibration and test sets using the project's standard approach.
    """
    # Get unique image IDs
    unique_imgs = torch.unique(img_ids)
    n_imgs = len(unique_imgs)
    
    # Create image mask
    img_mask = torch.zeros(n_imgs, dtype=torch.bool)
    img_mask[:] = True  # All images are relevant
    
    # Use random_split to get calibration/test split
    calib_mask, calib_img_idx, test_img_idx = random_split(
        img_mask, img_ids, calib_fraction, verbose=True
    )
    
    # Get indices for calibration and test
    cal_idx = torch.where(calib_mask)[0]
    test_idx = torch.where(~calib_mask)[0]
    
    # Split data
    cal_data = (
        features[cal_idx],
        pred_coords[cal_idx],
        gt_coords[cal_idx],
        errors[cal_idx]
    )
    
    test_data = (
        features[test_idx],
        pred_coords[test_idx],
        gt_coords[test_idx],
        errors[test_idx]
    )
    
    return cal_data, test_data


def train_model(train_features, train_pred, train_gt, cal_data, test_data, args, logger):
    """Train the regression model."""
    device = torch.device(args.device)
    
    # Create training dataset
    train_dataset = TensorDataset(train_features, train_pred, train_gt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Move calibration and test data to device
    cal_features, cal_pred, cal_gt, cal_errors = cal_data
    cal_data = (
        cal_features.to(device),
        cal_pred.to(device),
        cal_gt.to(device),
        cal_errors.to(device)
    )
    
    test_features, test_pred, test_gt, test_errors = test_data
    test_data = (
        test_features.to(device),
        test_pred.to(device),
        test_gt.to(device),
        test_errors.to(device)
    )
    
    # Initialize model
    model = RegressionScoringFunction(
        input_dim=train_features.shape[1],
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01
    )
    
    # Loss function with lower efficiency weight initially
    criterion = RegressionCoverageLoss(
        target_coverage=args.target_coverage,
        efficiency_weight=0.001,  # Start very low, increase during training
        calibration_weight=args.calibration_weight
    )
    
    # Training history
    history = {
        'train_loss': [], 'test_loss': [], 'coverage': [],
        'efficiency': [], 'tau_values': [], 'calibration': []
    }
    
    best_coverage_gap = float('inf')
    best_epoch = 0
    
    logger.info(f"Starting training with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Train: {len(train_features)}, Cal: {len(cal_features)}, Test: {len(test_features)} samples")
    
    for epoch in range(args.num_epochs):
        # Use fixed tau = 1.0 (model learns appropriate widths)
        if epoch == 0:
            tau = torch.tensor(1.0, device=device)
            logger.info(f"Using fixed tau = {tau.item():.1f}")
        
        # Training
        model.train()
        train_losses = []
        
        for batch_features, batch_pred, batch_gt in train_loader:
            batch_features = batch_features.to(device)
            batch_pred = batch_pred.to(device)
            batch_gt = batch_gt.to(device)
            
            widths = model(batch_features)
            losses = criterion(widths, batch_gt, batch_pred, tau)
            
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            
            train_losses.append({k: v.item() for k, v in losses.items()})
        
        # Test set evaluation
        model.eval()
        with torch.no_grad():
            test_features, test_pred, test_gt, test_errors = test_data
            test_widths = model(test_features)
            test_losses = criterion(test_widths, test_gt, test_pred, tau)
            
            # Calculate metrics with CORRECT coverage definition
            interval_half_widths = test_widths * tau
            lower_bounds = test_pred - interval_half_widths.expand(-1, 4)
            upper_bounds = test_pred + interval_half_widths.expand(-1, 4)
            
            # Check if ground truth falls within intervals
            test_covered = ((test_gt >= lower_bounds) & (test_gt <= upper_bounds)).all(dim=1).float()
            test_coverage = test_covered.mean().item()
            avg_width = test_widths.mean().item()
            
            # Calibration metric (correlation between widths and errors)
            correlation = test_losses.get('correlation', torch.tensor(0.0)).item()
        
        scheduler.step()
        
        # Average training losses
        avg_train_losses = {
            k: np.mean([l[k] for l in train_losses]) 
            for k in train_losses[0].keys()
        }
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
            logger.info(f"  Train Loss: {avg_train_losses['total']:.4f}")
            logger.info(f"  Test Coverage: {test_coverage:.3f} (target: {args.target_coverage})")
            logger.info(f"  Avg Width: {avg_width:.3f}, Correlation: {correlation:.3f}")
        
        # Save history
        history['train_loss'].append(avg_train_losses['total'])
        history['test_loss'].append(test_losses['total'].item())
        history['coverage'].append(test_coverage)
        history['efficiency'].append(avg_width)
        history['tau_values'].append(tau.item())
        history['calibration'].append(correlation)
        
        # Update efficiency weight based on coverage
        if test_coverage >= 0.85:  # Close to target
            criterion.efficiency_weight = min(args.efficiency_weight, criterion.efficiency_weight * 1.05)
        elif test_coverage < 0.5:  # Far from target
            criterion.efficiency_weight = max(0.0001, criterion.efficiency_weight * 0.9)
        
        # Save best model based on coverage gap
        coverage_gap = abs(test_coverage - args.target_coverage)
        if coverage_gap < best_coverage_gap:
            best_coverage_gap = coverage_gap
            best_epoch = epoch
            save_regression_model(
                model, optimizer, epoch, test_losses, 
                tau.item(), args.output_dir / "best_model.pt"
            )
        
        # Early stopping
        if epoch - best_epoch > args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"\nTraining completed! Best epoch: {best_epoch+1}")
    logger.info(f"Best coverage gap: {best_coverage_gap:.3f}")
    
    return model, history


def plot_results(history, output_dir):
    """Plot training results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(history['test_loss'], label='Test', alpha=0.7)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coverage
    axes[0, 1].plot(history['coverage'], linewidth=2)
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Target')
    axes[0, 1].set_title('Test Set Coverage')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Interval width
    axes[0, 2].plot(history['efficiency'], color='green')
    axes[0, 2].set_title('Average Interval Width')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Tau values
    axes[1, 0].plot(history['tau_values'], color='orange')
    axes[1, 0].set_title('Tau Values (from Calibration Set)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calibration
    axes[1, 1].plot(history['calibration'], color='purple')
    axes[1, 1].set_title('Calibration STD')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Coverage vs Efficiency
    axes[1, 2].scatter(history['efficiency'], history['coverage'], 
                      c=range(len(history['coverage'])), cmap='viridis', alpha=0.6)
    axes[1, 2].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Average Width')
    axes[1, 2].set_ylabel('Coverage')
    axes[1, 2].set_title('Coverage vs Efficiency Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train regression-based scoring function")
    
    # Data configuration
    parser.add_argument('--config_file', type=str, default='cfg_std_rank')
    parser.add_argument('--config_path', type=str, default='config/coco_val/')
    parser.add_argument('--cache_dir', type=str, default='learnable_scoring_fn/experiments/cache')
    
    # Model architecture
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128, 64])
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    
    # Loss weights
    parser.add_argument('--target_coverage', type=float, default=0.9)
    parser.add_argument('--efficiency_weight', type=float, default=0.05)
    parser.add_argument('--calibration_weight', type=float, default=0.1)
    
    # Other parameters  
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--calib_fraction', type=float, default=0.5)
    parser.add_argument('--output_dir', type=Path, default=Path('learnable_scoring_fn/experiments/real_data_v1'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("="*60)
    logger.info("Regression-Based Scoring Function Training")
    logger.info("Using Project's Standard Data Splits:")
    logger.info("  - COCO train set for training")
    logger.info("  - COCO val set split into calibration/test")
    logger.info("="*60)
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # For faster development, use only validation set predictions
        # Split val set into train/cal/test portions
        logger.info("Using validation set for train/cal/test split (faster development)")
        val_img_list, val_ist_list = collect_predictions_for_dataset(
            args.config_file, args.config_path, 'val', args.cache_dir, logger
        )
        
        # Use val data for both training and testing
        train_img_list, train_ist_list = val_img_list, val_ist_list
        
        # Prepare training data
        train_features, train_pred, train_gt, train_errors, train_img_ids, \
            train_feat_ext, train_uncert_ext = prepare_regression_data(
                train_img_list, train_ist_list, logger
            )
        
        # Prepare validation data
        val_features, val_pred, val_gt, val_errors, val_img_ids, \
            val_feat_ext, val_uncert_ext = prepare_regression_data(
                val_img_list, val_ist_list, logger
            )
        
        # Normalize validation features using training statistics
        val_features_raw = torch.cat([
            val_feat_ext.extract_features(val_pred, val_gt[:, 0]),  # Using first coord as proxy for confidence
            val_uncert_ext.extract_uncertainty_features(val_pred, val_gt[:, 0])
        ], dim=1)
        val_features = train_feat_ext.normalize_features(val_features_raw)
        
        # Split validation data into calibration and test sets
        cal_data, test_data = split_val_data(
            val_features, val_pred, val_gt, val_errors, val_img_ids, 
            args.calib_fraction
        )
        
        # Save feature statistics
        torch.save({
            'feature_stats': train_feat_ext.feature_stats,
            'error_stats': train_uncert_ext.error_stats
        }, args.output_dir / 'data_stats.pt')
        
        # Train model
        model, history = train_model(
            train_features, train_pred, train_gt, 
            cal_data, test_data, args, logger
        )
        
        # Plot results
        plot_results(history, args.output_dir)
        
        # Save final results
        results = {
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'history': history,
            'data_stats': {
                'train_samples': len(train_features),
                'cal_samples': len(cal_data[0]),
                'test_samples': len(test_data[0]),
                'feature_dim': train_features.shape[1]
            },
            'final_metrics': {
                'coverage': history['coverage'][-1],
                'avg_width': history['efficiency'][-1],
                'tau': history['tau_values'][-1],
                'calibration_std': history['calibration'][-1]
            }
        }
        
        with open(args.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to {args.output_dir}")
        logger.info("Data splits used:")
        logger.info(f"  - Training: {len(train_features)} samples from COCO train")
        logger.info(f"  - Calibration: {len(cal_data[0])} samples from COCO val")
        logger.info(f"  - Test: {len(test_data[0])} samples from COCO val")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()