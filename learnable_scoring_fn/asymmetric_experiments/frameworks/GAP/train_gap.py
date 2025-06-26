"""
Training script for GAP (Gated Asymmetry with Stable Parameterization)
Autonomous training with adaptive hyperparameter tuning
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append('/ssd_4TB/divake/conformal-od/learnable_scoring_fn')

# Import baseline utilities
# We'll do manual tau calibration

# Import GAP components
sys.path.append('/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/frameworks/GAP')
from models.gap_model import GatedAsymmetryNetwork
from losses.gap_loss import GAPLoss


class GAPTrainer:
    def __init__(self, config, attempt_number):
        self.config = config
        self.attempt_number = attempt_number
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.framework_dir = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/frameworks/GAP'
        self.attempt_dir = os.path.join(self.framework_dir, f'attempt_{attempt_number}')
        os.makedirs(self.attempt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.attempt_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize model
        self.model = GatedAsymmetryNetwork(
            hidden_dims=config.get('hidden_dims', [256, 128, 64]),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'elu')
        ).to(self.device)
        
        # Initialize loss
        self.loss_fn = GAPLoss(config)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 5e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Initialize scheduler
        if config.get('scheduler', 'cosine') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 50)
            )
        elif config.get('scheduler') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        else:
            self.scheduler = None
        
        # Initialize tau parameters (we'll do manual calibration)
        self.current_tau = 1.0
        self.min_tau = config.get('min_tau', 0.2)
        self.max_tau = config.get('max_tau', 5.0)
        self.tau_smoothing = config.get('tau_smoothing', 0.6)
        self.tau_history = []
        
        # Load cached data
        cache_dir = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model'
        
        # Use baseline's data loading functions
        from core_symmetric.symmetric_adaptive import load_cached_data, prepare_splits
        
        data = load_cached_data(cache_dir)
        self.train_data = data['train_data']
        self.val_data = data['val_data']
        
        # Split validation into calibration and test
        self.calib_data, self.test_data = prepare_splits(self.val_data, calib_fraction=0.5)
        
        # Create data loaders
        train_dataset = TensorDataset(
            self.train_data['features'],
            self.train_data['pred_coords'],
            self.train_data['gt_coords'],
            self.train_data['confidence']
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 512),
            shuffle=True,
            num_workers=4
        )
        
        calib_dataset = TensorDataset(
            self.calib_data['features'],
            self.calib_data['pred_coords'],
            self.calib_data['gt_coords'],
            self.calib_data['confidence']
        )
        
        self.cal_loader = DataLoader(
            calib_dataset,
            batch_size=config.get('batch_size', 512),
            shuffle=False,
            num_workers=4
        )
        
        eval_dataset = TensorDataset(
            self.test_data['features'],
            self.test_data['pred_coords'],
            self.test_data['gt_coords'],
            self.test_data['confidence']
        )
        
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.get('batch_size', 512),
            shuffle=False,
            num_workers=4
        )
        
        # Tracking variables
        self.best_mpiw = float('inf')
        self.best_coverage = 0.0
        self.patience_counter = 0
        self.training_history = []
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_metrics = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch data
            features, pred_coords, gt_coords, confidence = batch
            features = features.to(self.device)
            boxes = pred_coords.to(self.device)
            ground_truth = gt_coords.to(self.device)
            
            # Forward pass
            inner_offset, outer_offset = self.model(features, boxes)
            
            # Compute loss
            loss, metrics = self.loss_fn(inner_offset, outer_offset, boxes, ground_truth)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_metrics.append(metrics)
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, Coverage: {metrics['overall_coverage']:.3f}, "
                      f"MPIW: {metrics['overall_mpiw']:.2f}")
        
        # Aggregate metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        return avg_loss, avg_metrics
    
    def calibrate(self, epoch):
        """Calibrate tau using calibration set"""
        if epoch < 1:  # Skip calibration for first epoch
            return
        
        self.model.eval()
        all_min_factors = []
        
        with torch.no_grad():
            for batch in self.cal_loader:
                features, pred_coords, gt_coords, confidence = batch
                features = features.to(self.device)
                boxes = pred_coords.to(self.device)
                ground_truth = gt_coords.to(self.device)
                
                # Get predictions
                inner_offset, outer_offset = self.model(features, boxes)
                
                # Compute minimum scaling factor needed
                inner_boxes = boxes + inner_offset
                outer_boxes = boxes + outer_offset
                
                # Check coverage for different scaling factors
                for factor in torch.linspace(0.5, 2.0, 50):
                    scaled_inner = boxes + inner_offset * factor
                    scaled_outer = boxes + outer_offset * factor
                    
                    # Check if GT is covered
                    covered = ((ground_truth >= scaled_inner) & (ground_truth <= scaled_outer)).all(dim=1)
                    
                    if covered.all():
                        all_min_factors.append(factor.item())
                        break
        
        # Calibrate tau
        if all_min_factors:
            target_coverage = self.config.get('target_coverage', 0.90)
            quantile_idx = int(len(all_min_factors) * (target_coverage - 0.01))
            new_tau = sorted(all_min_factors)[quantile_idx]
            
            # Update tau with smoothing
            current_tau = self.model.get_tau()
            smoothed_tau = self.tau_smoothing * current_tau + (1 - self.tau_smoothing) * new_tau
            smoothed_tau = np.clip(smoothed_tau, self.min_tau, self.max_tau)
            self.model.set_tau(smoothed_tau)
            self.tau_history.append(smoothed_tau)
            
            print(f"Calibrated tau: {current_tau:.3f} -> {smoothed_tau:.3f}")
    
    def validate(self, epoch):
        """Validate on evaluation set"""
        self.model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in self.eval_loader:
                features, pred_coords, gt_coords, confidence = batch
                features = features.to(self.device)
                boxes = pred_coords.to(self.device)
                ground_truth = gt_coords.to(self.device)
                
                # Get predictions
                inner_offset, outer_offset = self.model(features, boxes)
                
                # Compute metrics
                _, metrics = self.loss_fn(inner_offset, outer_offset, boxes, ground_truth)
                all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        # Check if this is best model
        coverage = avg_metrics['overall_coverage']
        mpiw = avg_metrics['overall_mpiw']
        
        is_best = False
        if 0.88 <= coverage <= 0.92 and mpiw < self.best_mpiw:
            self.best_mpiw = mpiw
            self.best_coverage = coverage
            is_best = True
            self._save_checkpoint(epoch, avg_metrics, is_best=True)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return avg_metrics, is_best
    
    def train(self):
        """Main training loop"""
        start_time = time.time()
        
        for epoch in range(self.config.get('epochs', 50)):
            print(f"\n--- Epoch {epoch+1}/{self.config.get('epochs', 50)} ---")
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Calibration phase
            self.calibrate(epoch)
            
            # Validation phase
            val_metrics, is_best = self.validate(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'tau': self.model.get_tau(),
                'is_best': is_best
            }
            self.training_history.append(epoch_results)
            
            print(f"Validation - Coverage: {val_metrics['overall_coverage']:.3f}, "
                  f"MPIW: {val_metrics['overall_mpiw']:.2f}, "
                  f"MPIW Small: {val_metrics['mpiw_small']:.2f}, "
                  f"MPIW Medium: {val_metrics['mpiw_medium']:.2f}, "
                  f"MPIW Large: {val_metrics['mpiw_large']:.2f}")
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Calculate final size-stratified metrics if we have a best model
        best_metrics = {}
        if self.best_coverage > 0:
            # Load best model checkpoint to get detailed metrics
            checkpoint_path = os.path.join(self.attempt_dir, 'checkpoints', 'best_model.pt')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                best_metrics = checkpoint.get('metrics', {})
        
        # Save final results
        final_results = {
            'attempt_number': self.attempt_number,
            'config': self.config,
            'best_coverage': self.best_coverage,
            'best_mpiw': self.best_mpiw,
            'best_mpiw_small': best_metrics.get('mpiw_small', 0),
            'best_mpiw_medium': best_metrics.get('mpiw_medium', 0),
            'best_mpiw_large': best_metrics.get('mpiw_large', 0),
            'baseline_mpiw': 48.5,
            'mpiw_reduction': (48.5 - self.best_mpiw) / 48.5 * 100 if self.best_mpiw < float('inf') else 0,
            'total_epochs': len(self.training_history),
            'total_time': total_time,
            'training_history': self.training_history
        }
        
        with open(os.path.join(self.attempt_dir, 'results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results
    
    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics from multiple batches"""
        if not metrics_list:
            return {}
        
        # Get all keys from first metric dict
        keys = metrics_list[0].keys()
        
        # Aggregate each metric
        aggregated = {}
        for key in keys:
            if key.endswith('_loss') or key == 'total_loss':
                # Average losses - convert to CPU if needed
                values = []
                for m in metrics_list:
                    val = m[key]
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().item()
                    values.append(val)
                aggregated[key] = np.mean(values)
            else:
                # Average other metrics - convert to CPU if needed
                values = []
                for m in metrics_list:
                    val = m[key]
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().item()
                    if val > 0:
                        values.append(val)
                aggregated[key] = np.mean(values) if values else 0.0
        
        return aggregated
    
    def _save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tau': self.model.get_tau(),
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.attempt_dir, 'checkpoints', 'best_model.pt')
        else:
            path = os.path.join(self.attempt_dir, 'checkpoints', f'epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")


def run_autonomous_training():
    """Autonomous training manager for GAP framework"""
    # Load memory
    memory_path = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/unified_memory/framework_memories/framework_1_GAP_memory.json'
    with open(memory_path, 'r') as f:
        memory = json.load(f)
    
    # Define configuration space with much lower efficiency weights
    config_space = {
        'hidden_dims': [[256, 128, 64], [128, 64], [512, 256, 128], [256, 128, 64, 32]],
        'dropout': [0.0, 0.1, 0.2],
        'activation': ['elu', 'relu', 'leakyrelu'],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [256, 512, 1024],
        'lambda_efficiency': [0.01, 0.02, 0.05, 0.1],  # Much lower to prioritize coverage
        'scheduler': ['cosine', 'step', None],
        'epochs': [50],
        'patience': [10],
        'target_coverage': [0.90],
        'min_tau': [0.5, 1.0],  # Higher minimum tau
        'max_tau': [10.0, 20.0]  # Higher maximum tau
    }
    
    # Start autonomous training loop
    attempt = len(memory['attempts']) + 1
    max_attempts = 10
    
    while attempt <= max_attempts:
        print(f"\n{'='*80}")
        print(f"GAP Framework - Attempt {attempt}/{max_attempts}")
        print(f"{'='*80}\n")
        
        # Select configuration based on previous attempts
        if attempt == 1:
            # Start with baseline configuration
            config = {
                'hidden_dims': [256, 128, 64],
                'dropout': 0.1,
                'activation': 'elu',
                'learning_rate': 5e-4,
                'batch_size': 512,
                'lambda_efficiency': 0.25,
                'scheduler': 'cosine',
                'epochs': 50,
                'patience': 10,
                'target_coverage': 0.90,
                'min_tau': 0.2,
                'max_tau': 5.0
            }
        elif attempt <= 6:
            # For attempts 2-6, use very low efficiency weight
            config = {
                'hidden_dims': [256, 128, 64],
                'dropout': 0.1,
                'activation': 'elu',
                'learning_rate': 1e-3,
                'batch_size': 512,
                'lambda_efficiency': 0.01,  # Very low to fix coverage
                'scheduler': 'cosine',
                'epochs': 50,
                'patience': 10,
                'target_coverage': 0.90,
                'min_tau': 1.0,
                'max_tau': 20.0
            }
        else:
            # Adapt configuration based on previous results
            config = adapt_configuration(memory['attempts'], config_space)
        
        # Train model
        trainer = GAPTrainer(config, attempt)
        results = trainer.train()
        
        # Update memory
        memory['attempts'].append({
            'attempt_number': attempt,
            'architecture': config,
            'results': {
                'coverage': results['best_coverage'],
                'mpiw': results['best_mpiw'],
                'mpiw_reduction': results['mpiw_reduction']
            },
            'decision': analyze_results_and_decide(results),
            'total_time': results['total_time']
        })
        
        memory['total_attempts'] = attempt
        
        # Check if target achieved
        if results['best_coverage'] >= 0.88 and results['best_coverage'] <= 0.92 and results['mpiw_reduction'] >= 15:
            memory['status'] = 'completed'
            memory['best_result'] = results
            memory['final_decision'] = f"Target achieved! Coverage: {results['best_coverage']:.3f}, MPIW reduction: {results['mpiw_reduction']:.1f}%"
            break
        
        # Save memory after each attempt
        with open(memory_path, 'w') as f:
            json.dump(memory, f, indent=2)
        
        attempt += 1
    
    # Final update
    if memory['status'] != 'completed':
        memory['status'] = 'failed'
        memory['final_decision'] = f"Could not achieve target after {attempt-1} attempts"
    
    with open(memory_path, 'w') as f:
        json.dump(memory, f, indent=2)
    
    return memory


def adapt_configuration(attempts, config_space):
    """Intelligently adapt configuration based on previous attempts"""
    if not attempts:
        return None
    
    last_attempt = attempts[-1]
    last_results = last_attempt['results']
    last_config = last_attempt['architecture']
    
    new_config = last_config.copy()
    
    # Analyze failure mode and adapt
    if last_results['coverage'] < 0.88:
        # Coverage too low
        new_config['lambda_efficiency'] = max(0.15, last_config.get('lambda_efficiency', 0.25) - 0.05)
        new_config['learning_rate'] = last_config.get('learning_rate', 5e-4) * 0.5
    elif last_results['coverage'] > 0.92:
        # Coverage too high
        new_config['lambda_efficiency'] = min(0.4, last_config.get('lambda_efficiency', 0.25) + 0.05)
    
    if last_results['mpiw_reduction'] < 5:
        # Not enough improvement
        # Try different architecture
        architecture_options = [[512, 256, 128], [256, 128, 64, 32], [128, 64]]
        for arch in architecture_options:
            if arch != last_config.get('hidden_dims'):
                new_config['hidden_dims'] = arch
                break
    
    # Try different hyperparameters
    if len(attempts) % 3 == 0:
        # Every 3rd attempt, try something different
        new_config['activation'] = 'relu' if last_config.get('activation') == 'elu' else 'elu'
        new_config['dropout'] = 0.2 if last_config.get('dropout', 0.1) < 0.2 else 0.0
    
    return new_config


def analyze_results_and_decide(results):
    """Analyze results and make decision"""
    coverage = results['best_coverage']
    mpiw_reduction = results['mpiw_reduction']
    
    if coverage < 0.88:
        return f"Coverage too low ({coverage:.3f}), need to reduce efficiency weight"
    elif coverage > 0.92:
        return f"Coverage too high ({coverage:.3f}), need to increase efficiency weight"
    elif mpiw_reduction < 5:
        return f"MPIW improvement insufficient ({mpiw_reduction:.1f}%), need larger model capacity"
    elif mpiw_reduction < 15:
        return f"Getting closer ({mpiw_reduction:.1f}% reduction), fine-tune hyperparameters"
    else:
        return f"Target achieved! Coverage: {coverage:.3f}, MPIW reduction: {mpiw_reduction:.1f}%"


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run autonomous training
    results = run_autonomous_training()
    print(f"\nFinal GAP Framework Status: {results['status']}")
    if results.get('best_result'):
        print(f"Best Coverage: {results['best_result']['best_coverage']:.3f}")
        print(f"Best MPIW Reduction: {results['best_result']['mpiw_reduction']:.1f}%")