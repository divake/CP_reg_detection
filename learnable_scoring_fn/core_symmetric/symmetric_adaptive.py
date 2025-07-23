"""Main training script for symmetric adaptive conformal prediction."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json
import pickle
from datetime import datetime

# Import our modules
from .models.symmetric_mlp import SymmetricAdaptiveMLP
from .losses.symmetric_loss import SymmetricAdaptiveLoss
from .calibration.tau_calibration import TauCalibrator
from .utils.logging import AdaptiveConformalLogger
from .utils.visualization import plot_training_results, plot_tau_evolution

# Import feature extraction from parent
import sys
sys.path.append(str(Path(__file__).parent.parent))
from feature_utils import FeatureExtractor


def load_cached_data(cache_dir: str) -> Dict[str, Any]:
    """Load cached features and predictions."""
    cache_path = Path(cache_dir)
    
    print(f"Loading cached data from {cache_path}")
    
    # Load features (these are already dictionaries)
    train_data = torch.load(cache_path / "features_train.pt")
    val_data = torch.load(cache_path / "features_val.pt")
    
    # Load predictions
    with open(cache_path / "predictions_train.pkl", 'rb') as f:
        train_preds = pickle.load(f)
    
    with open(cache_path / "predictions_val.pkl", 'rb') as f:
        val_preds = pickle.load(f)
    
    # Extract features from the dictionaries
    train_features = train_data['features']
    val_features = val_data['features']
    
    print(f"Loaded train features: {train_features.shape}")
    print(f"Loaded val features: {val_features.shape}")
    
    return {
        'train_features': train_features,
        'train_data': train_data,  # Full data dict
        'val_features': val_features,
        'val_data': val_data,      # Full data dict
        'train_predictions': train_preds,
        'val_predictions': val_preds
    }


def prepare_splits(
    val_data: Dict[str, torch.Tensor],
    calib_fraction: float = 0.5,
    seed: int = 42
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split validation data into calibration and test sets."""
    torch.manual_seed(seed)
    
    # Extract data from validation dictionary
    val_features = val_data['features']
    pred_coords = val_data['pred_coords']
    gt_coords = val_data['gt_coords']
    confidence = val_data['confidence']
    
    print(f"Total validation samples: {len(val_features)}")
    
    # Split indices
    n_val = len(val_features)
    n_calib = int(n_val * calib_fraction)
    
    indices = torch.randperm(n_val)
    calib_idx = indices[:n_calib]
    test_idx = indices[n_calib:]
    
    # Create splits
    calib_data = {
        'features': val_features[calib_idx],
        'pred_coords': pred_coords[calib_idx],
        'gt_coords': gt_coords[calib_idx],
        'confidence': confidence[calib_idx]
    }
    
    test_data = {
        'features': val_features[test_idx],
        'pred_coords': pred_coords[test_idx],
        'gt_coords': gt_coords[test_idx],
        'confidence': confidence[test_idx]
    }
    
    print(f"Calibration samples: {len(calib_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    return calib_data, test_data


def compute_size_stratified_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    tau: float,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """Compute metrics stratified by object size."""
    model.eval()
    
    # Size bins (matching COCO standards)
    size_bins = {
        'small': (0, 32**2),
        'medium': (32**2, 96**2),
        'large': (96**2, float('inf'))
    }
    
    # Initialize collectors
    results = {cat: {'covered': [], 'mpiw': []} for cat in size_bins}
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch[0].to(device)
            pred_coords = batch[1].to(device)
            gt_coords = batch[2].to(device)
            
            # Get predictions
            widths = model(features)
            scaled_widths = widths * tau
            
            # Check coverage
            lower = pred_coords - scaled_widths
            upper = pred_coords + scaled_widths
            covered = ((gt_coords >= lower) & (gt_coords <= upper)).all(dim=1)
            
            # Compute MPIW
            mpiw = (2 * scaled_widths).mean(dim=1)
            
            # Compute object sizes
            box_widths = gt_coords[:, 2] - gt_coords[:, 0]
            box_heights = gt_coords[:, 3] - gt_coords[:, 1]
            areas = box_widths * box_heights
            
            # Stratify
            for i in range(len(areas)):
                area = areas[i].item()
                for cat, (min_size, max_size) in size_bins.items():
                    if min_size <= area < max_size:
                        results[cat]['covered'].append(covered[i].item())
                        results[cat]['mpiw'].append(mpiw[i].item())
                        break
    
    # Aggregate results
    final_results = {}
    for cat, data in results.items():
        if data['covered']:
            final_results[cat] = {
                'coverage': np.mean(data['covered']),
                'mpiw': np.mean(data['mpiw']),
                'count': len(data['covered'])
            }
        else:
            final_results[cat] = {
                'coverage': 0.0,
                'mpiw': 0.0,
                'count': 0
            }
    
    return final_results


def train_symmetric_adaptive(
    config: Dict,
    cache_dir: str,
    output_dir: str,
    log_dir: Optional[str] = None,
    device: Optional[str] = None
):
    """
    Main training function for symmetric adaptive conformal prediction.
    
    Args:
        config: Training configuration
        cache_dir: Directory with cached features/predictions
        output_dir: Directory to save models
        log_dir: Directory for logs
    """
    # Use the output_dir directly - don't create nested directories
    experiment_dir = Path(output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract experiment name from the directory name
    experiment_name = experiment_dir.name
    
    # Create subdirectories
    model_dir = experiment_dir / "models"
    model_dir.mkdir(exist_ok=True)
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Initialize logger with proper directory
    logger = AdaptiveConformalLogger(log_dir, experiment_name)
    
    # Set device
    if device is None:
        device = config.get('experiment', {}).get('device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load cached data
    data = load_cached_data(cache_dir)
    
    # Use the pre-processed data from cache
    train_features = data['train_features']
    train_pred_coords = data['train_data']['pred_coords']
    train_gt_coords = data['train_data']['gt_coords']
    
    # Prepare calibration and test splits
    calib_data, test_data = prepare_splits(
        data['val_data'],
        calib_fraction=0.5
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        train_features, train_pred_coords, train_gt_coords
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    calib_dataset = TensorDataset(
        calib_data['features'],
        calib_data['pred_coords'],
        calib_data['gt_coords']
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    test_dataset = TensorDataset(
        test_data['features'],
        test_data['pred_coords'],
        test_data['gt_coords']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Initialize model
    feature_dim = train_features.shape[1]
    model_config = config['model']['architecture']
    model = SymmetricAdaptiveMLP(
        input_dim=feature_dim,
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config['dropout_rate'],
        activation=model_config['activation'],
        use_batch_norm=model_config['use_batch_norm']
    ).to(device)
    
    print(f"Model: {model.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize loss function
    loss_config = config['loss']
    if loss_config.get('use_size_aware_loss', False):
        from .losses.size_aware_loss import SizeAwareSymmetricLoss
        criterion = SizeAwareSymmetricLoss(
            small_target_coverage=loss_config['size_targets']['small'],
            medium_target_coverage=loss_config['size_targets']['medium'],
            large_target_coverage=loss_config['size_targets']['large'],
            lambda_efficiency=loss_config['lambda_efficiency'],
            coverage_loss_type=loss_config['coverage_loss_type'],
            size_normalization=loss_config['size_normalization'],
            small_threshold=loss_config['size_thresholds']['small'],
            large_threshold=loss_config['size_thresholds']['large']
        )
        print("Using SizeAwareSymmetricLoss with targets:")
        print(f"  Small objects (<{loss_config['size_thresholds']['small']}): {loss_config['size_targets']['small']:.0%}")
        print(f"  Medium objects: {loss_config['size_targets']['medium']:.0%}")
        print(f"  Large objects (>{loss_config['size_thresholds']['large']}): {loss_config['size_targets']['large']:.0%}")
    else:
        criterion = SymmetricAdaptiveLoss(
            target_coverage=config['calibration']['target_coverage'],
            lambda_efficiency=loss_config['lambda_efficiency'],
            coverage_loss_type=loss_config['coverage_loss_type'],
            size_normalization=loss_config.get('size_normalization', True)
        )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    lr_scheduler_config = config['training']['lr_scheduler']
    scheduler_type = lr_scheduler_config['type']
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=lr_scheduler_config['min_lr']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler_config.get('step_size', 10),
            gamma=lr_scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=lr_scheduler_config.get('decay_rate', 0.95)
        )
    else:
        scheduler = None
    
    # Initialize tau calibrator  
    calib_config = config['calibration']
    tau_calibrator = TauCalibrator(
        target_coverage=calib_config['target_coverage'],
        min_tau=calib_config['min_tau'],
        max_tau=calib_config['max_tau'],
        smoothing_factor=calib_config['tau_smoothing']
    )
    
    # Training state
    current_tau = 1.0
    best_coverage_error = float('inf')
    best_mpiw = float('inf')
    history = {}
    
    # Training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        logger.log_epoch_start(epoch, current_tau)
        
        # Phase 1: Training
        model.train()
        train_losses = []
        train_metrics = {}
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch[0].to(device)
            pred_coords = batch[1].to(device)
            gt_coords = batch[2].to(device)
            
            # Forward pass
            widths = model(features)
            
            # Compute loss
            loss_dict = criterion(
                pred_coords, gt_coords, widths, current_tau
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.get('grad_clip_norm', 1.0)
            )
            
            optimizer.step()
            
            # Log batch metrics
            train_losses.append(loss_dict['total'].item())
            
            if batch_idx % 100 == 0:
                logger.log_training_phase(epoch, batch_idx, loss_dict)
        
        # Aggregate training metrics
        train_metrics['total'] = np.mean(train_losses)
        logger.log_epoch_metrics(epoch, 'train', train_metrics)
        
        # Save training loss to history
        if 'train_loss' not in history:
            history['train_loss'] = []
        history['train_loss'].append(train_metrics['total'])
        
        # Phase 2: Calibration (skip for epoch 1)
        if epoch > 1:
            old_tau = current_tau
            current_tau, calib_stats = tau_calibrator.calibrate(
                model, calib_data, config['training']['batch_size'], device
            )
            logger.log_calibration_phase(epoch, old_tau, current_tau, calib_stats)
        
        # Phase 3: Validation
        model.eval()
        val_losses = []
        val_coverages = []
        val_mpiws = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch[0].to(device)
                pred_coords = batch[1].to(device)
                gt_coords = batch[2].to(device)
                
                # Get predictions
                widths = model(features)
                
                # Compute metrics
                loss_dict = criterion(
                    pred_coords, gt_coords, widths, current_tau
                )
                
                val_losses.append(loss_dict['total'].item())
                val_coverages.append(loss_dict['coverage_rate'].item())
                val_mpiws.append(loss_dict['avg_mpiw'].item())
        
        # Aggregate validation metrics
        val_metrics = {
            'total': np.mean(val_losses),
            'coverage_rate': np.mean(val_coverages),
            'avg_mpiw': np.mean(val_mpiws),
            'tau': current_tau
        }
        
        # Compute size-stratified metrics
        size_metrics = compute_size_stratified_metrics(
            model, test_loader, current_tau, device
        )
        
        logger.log_validation_phase(epoch, val_metrics, size_metrics)
        
        # Smart model checkpointing
        coverage_error = abs(val_metrics['coverage_rate'] - config['calibration']['target_coverage'])
        min_target_coverage = config['calibration']['min_coverage']
        max_target_coverage = config['calibration']['max_coverage']
        
        # Save best model logic - prioritize coverage in target range with lowest MPIW
        save_model = False
        save_reason = ""
        
        if min_target_coverage <= val_metrics['coverage_rate'] <= max_target_coverage:
            # In target range - prioritize by MPIW
            if val_metrics['avg_mpiw'] < best_mpiw:
                save_model = True
                save_reason = f"Target coverage ({val_metrics['coverage_rate']:.3f}) with better MPIW ({val_metrics['avg_mpiw']:.1f})"
                best_mpiw = val_metrics['avg_mpiw']
                best_coverage_error = coverage_error
            elif abs(val_metrics['avg_mpiw'] - best_mpiw) < 0.1 and coverage_error < best_coverage_error:
                save_model = True
                save_reason = f"Similar MPIW, closer to {config['target_coverage']:.0%} coverage"
                best_coverage_error = coverage_error
        
        if save_model:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_config': model.get_config(),
                'tau': current_tau,
                'metrics': val_metrics,
                'size_metrics': size_metrics,
                'config': config
            }
            # Save with descriptive filename
            model_filename = f'best_model_cov{val_metrics["coverage_rate"]:.3f}_mpiw{val_metrics["avg_mpiw"]:.1f}.pt'
            torch.save(checkpoint, model_dir / model_filename)
            # Also save as 'best_model.pt' for easy access
            torch.save(checkpoint, model_dir / 'best_model.pt')
            logger.log_best_model(epoch, save_reason, val_metrics)
        
        # Only save checkpoints when significant progress is made
        # No need to save every 10 epochs - just save when we have a good model
        # This reduces clutter and focuses on meaningful checkpoints
        
        # Update history
        for key, value in val_metrics.items():
            if key not in history:
                history[key] = []
            history[key].append(value)
        
        # Save size metrics to history
        if 'size_metrics' not in history:
            history['size_metrics'] = []
        history['size_metrics'].append(size_metrics)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
            history.setdefault('learning_rate', []).append(
                optimizer.param_groups[0]['lr']
            )
        
        # Visualization
        logger.create_visualization(epoch)
        
        # Early stopping check based on stable coverage
        if epoch > config['training'].get('warmup_epochs', 5):
            # Check if we have enough history
            if len(history.get('coverage_rate', [])) >= 10:
                recent_coverages = history['coverage_rate'][-10:]
                # Check if coverage is stable in target range
                min_cov = config['calibration']['min_coverage']
                max_cov = config['calibration']['max_coverage']
                if all(min_cov <= c <= max_cov for c in recent_coverages):
                    recent_mpiws = history['avg_mpiw'][-10:]
                    avg_coverage = np.mean(recent_coverages)
                    std_coverage = np.std(recent_coverages)
                    print(f"\nEarly stopping: Coverage stable at {avg_coverage:.3f} (±{std_coverage:.3f})")
                    print(f"Average MPIW over last 10 epochs: {np.mean(recent_mpiws):.2f}")
                    break
    
    # Final summary
    logger.save_final_summary()
    
    # Save final plots and CSV to experiment directory
    from .utils.visualization import create_all_plots
    create_all_plots(history, experiment_dir, model_name='symmetric_adaptive')
    
    # Save configuration for reproducibility
    import yaml
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Save final results summary
    final_results = {
        'experiment_name': experiment_name,
        'final_tau': current_tau,
        'final_coverage': history['coverage_rate'][-1] if 'coverage_rate' in history else 0,
        'final_mpiw': history['avg_mpiw'][-1] if 'avg_mpiw' in history else 0,
        'best_model_saved': (model_dir / 'best_model.pt').exists(),
        'total_epochs': epoch,
        'size_metrics': size_metrics if 'size_metrics' in locals() else None
    }
    
    with open(experiment_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final tau: {current_tau:.4f}")
    print(f"Final coverage: {val_metrics['coverage_rate']:.3f}")
    print(f"Final MPIW: {val_metrics['avg_mpiw']:.2f}")
    
    return model, current_tau, history


if __name__ == "__main__":
    # Default configuration
    config = {
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 256,
        "target_coverage": 0.9,
        "lambda_efficiency": 0.1,
        "tau_smoothing": 0.7,
        "lr_scheduler": "cosine",
        "warmup_epochs": 5,
        "hidden_dims": [128, 128],
        "dropout_rate": 0.1,
        "activation": "relu",
        "use_batch_norm": True,
        "coverage_loss_type": "smooth_l1",
        "size_normalization": True,
        "grad_clip_norm": 1.0,
        "weight_decay": 1e-4,
        "min_lr": 1e-6
    }
    
    # Run training
    train_symmetric_adaptive(config)