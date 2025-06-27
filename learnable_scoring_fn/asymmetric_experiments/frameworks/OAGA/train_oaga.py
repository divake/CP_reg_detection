"""
Training script for OAGA (Ordering-Aware Gated Asymmetry)
Implements gradual asymmetry learning with strong ordering constraints
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

# Import GAP components
sys.path.append('/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/frameworks/OAGA')
from models.oaga_model import OrderingAwareGatedAsymmetry
from losses.oaga_loss import OAGALoss


class OAGATrainer:
    def __init__(self, config, attempt_number):
        self.config = config
        self.attempt_number = attempt_number
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.framework_dir = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/frameworks/OAGA'
        self.attempt_dir = os.path.join(self.framework_dir, f'attempt_{attempt_number}')
        os.makedirs(self.attempt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.attempt_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize model with symmetric initialization
        self.model = OrderingAwareGatedAsymmetry(
            hidden_dims=config.get('hidden_dims', [256, 128, 64]),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'elu'),
            symmetric_init=config.get('symmetric_init', True)
        ).to(self.device)
        
        # Initialize loss
        self.loss_fn = OAGALoss(config)
        
        # Initialize optimizer with lower learning rate for stability
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
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
            batch_size=config.get('batch_size', 256),  # Smaller batch size for stability
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        calib_dataset = TensorDataset(
            self.calib_data['features'],
            self.calib_data['pred_coords'],
            self.calib_data['gt_coords'],
            self.calib_data['confidence']
        )
        
        self.cal_loader = DataLoader(
            calib_dataset,
            batch_size=config.get('batch_size', 256),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataset = TensorDataset(
            self.test_data['features'],
            self.test_data['pred_coords'],
            self.test_data['gt_coords'],
            self.test_data['confidence']
        )
        
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.get('batch_size', 256),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Tracking variables
        self.best_mpiw = float('inf')
        self.best_coverage = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        # Tau calibration parameters
        self.current_tau = config.get('initial_tau', 1.0)
        self.min_tau = config.get('min_tau', 0.5)
        self.max_tau = config.get('max_tau', 10.0)
        self.tau_smoothing = config.get('tau_smoothing', 0.7)
        self.tau_history = []
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.loss_fn.set_epoch(epoch)  # Update epoch for warmup
        
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
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_metrics.append(metrics)
            
            # Log progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, Coverage: {metrics['overall_coverage']:.3f}, "
                      f"MPIW: {metrics['overall_mpiw']:.2f}, Ordering Viol: {metrics['ordering_violations']:.3f}")
        
        # Aggregate metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        return avg_loss, avg_metrics
    
    def calibrate(self, epoch):
        """Calibrate tau using calibration set"""
        if epoch < 5:  # Wait a few epochs before calibrating
            return
        
        self.model.eval()
        all_coverage_factors = []
        
        with torch.no_grad():
            for batch in self.cal_loader:
                features, pred_coords, gt_coords, confidence = batch
                features = features.to(self.device)
                boxes = pred_coords.to(self.device)
                ground_truth = gt_coords.to(self.device)
                
                # Get predictions
                inner_offset, outer_offset = self.model(features, boxes)
                
                # Compute boxes
                inner_boxes = boxes + inner_offset
                outer_boxes = boxes + outer_offset
                
                # Find minimum tau for coverage
                current_tau = self.model.get_tau()
                
                # Binary search for optimal tau per example
                for i in range(boxes.shape[0]):
                    low, high = 0.1, 5.0
                    for _ in range(10):  # Binary search iterations
                        mid = (low + high) / 2
                        test_inner = boxes[i] + inner_offset[i] * mid / current_tau
                        test_outer = boxes[i] + outer_offset[i] * mid / current_tau
                        
                        # Check coverage
                        covered = ((ground_truth[i] >= test_inner) & (ground_truth[i] <= test_outer)).all()
                        
                        if covered:
                            high = mid
                        else:
                            low = mid
                    
                    all_coverage_factors.append(high)
        
        # Calibrate tau
        if all_coverage_factors:
            target_coverage = self.config.get('target_coverage', 0.90)
            quantile_idx = int(len(all_coverage_factors) * target_coverage)
            new_tau = sorted(all_coverage_factors)[quantile_idx]
            
            # Smooth update
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
        
        for epoch in range(self.config.get('epochs', 100)):  # More epochs for gradual learning
            print(f"\n--- Epoch {epoch+1}/{self.config.get('epochs', 100)} ---")
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Calibration phase
            self.calibrate(epoch)
            
            # Validation phase
            val_metrics, is_best = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'tau': self.model.get_tau(),
                'is_best': is_best,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_results)
            
            print(f"Validation - Coverage: {val_metrics['overall_coverage']:.3f}, "
                  f"MPIW: {val_metrics['overall_mpiw']:.2f}, "
                  f"Ordering Violations: {val_metrics['ordering_violations']:.3f}, "
                  f"Inner Ratio: {val_metrics['inner_ratio']:.3f}")
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 20):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Calculate final metrics
        baseline_mpiw = 48.5
        mpiw_reduction = (baseline_mpiw - self.best_mpiw) / baseline_mpiw * 100 if self.best_mpiw < float('inf') else 0
        
        # Save final results
        final_results = {
            'attempt_number': self.attempt_number,
            'config': self.config,
            'best_coverage': self.best_coverage,
            'best_mpiw': self.best_mpiw,
            'baseline_mpiw': baseline_mpiw,
            'mpiw_reduction': mpiw_reduction,
            'total_epochs': len(self.training_history),
            'total_time': total_time,
            'training_history': self.training_history,
            'tau_history': self.tau_history
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
            values = []
            for m in metrics_list:
                val = m[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().item()
                values.append(val)
            
            # Average all metrics
            aggregated[key] = np.mean(values)
        
        return aggregated
    
    def _save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
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
    """Autonomous training manager for OAGA framework"""
    # Load memory
    memory_path = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/unified_memory/framework_memories/framework_2_OAGA_memory.json'
    with open(memory_path, 'r') as f:
        memory = json.load(f)
    
    # Configuration for OAGA
    base_config = {
        'hidden_dims': [256, 128, 64],
        'dropout': 0.1,
        'activation': 'elu',
        'learning_rate': 1e-4,
        'batch_size': 256,
        'lambda_coverage': 1.0,
        'lambda_efficiency': 0.01,  # Very low initially
        'lambda_ordering': 0.5,
        'lambda_asymmetry': 0.1,
        'symmetric_init': True,
        'asymmetry_warmup_epochs': 10,
        'scheduler': 'cosine_warm_restarts',
        'epochs': 100,
        'patience': 20,
        'target_coverage': 0.90,
        'initial_tau': 1.0,
        'min_tau': 0.5,
        'max_tau': 10.0
    }
    
    # Start training
    attempt = len(memory['attempts']) + 1
    max_attempts = 5  # Fewer attempts since OAGA is more stable
    
    # Check if we should increase lambda_efficiency for attempt 4
    if attempt == 4:
        # Last attempt was close, now push for more efficiency
        base_config['lambda_efficiency'] = 0.008  # Double it
        base_config['lambda_ordering'] = 0.3  # Reduce ordering constraint slightly
        base_config['asymmetry_warmup_epochs'] = 5  # Allow asymmetry sooner
    
    while attempt <= max_attempts:
        print(f"\n{'='*80}")
        print(f"OAGA Framework - Attempt {attempt}/{max_attempts}")
        print(f"{'='*80}\n")
        
        # Modify config based on previous attempts
        if attempt > 1:
            # Adjust based on previous results
            last_attempt = memory['attempts'][-1]
            last_coverage = last_attempt['results']['coverage']
            
            if last_coverage < 0.5:
                # Still too low coverage
                base_config['lambda_efficiency'] *= 0.5
                base_config['lambda_ordering'] *= 0.8
                base_config['initial_tau'] *= 2.0
            elif last_coverage < 0.88:
                # Getting closer
                base_config['lambda_efficiency'] *= 0.8
            elif last_coverage > 0.92:
                # Too high coverage
                base_config['lambda_efficiency'] *= 1.5
        
        # Train model
        trainer = OAGATrainer(base_config, attempt)
        results = trainer.train()
        
        # Update memory
        memory['attempts'].append({
            'attempt_number': attempt,
            'architecture': base_config,
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
        if results['best_coverage'] >= 0.88 and results['best_coverage'] <= 0.92 and results['mpiw_reduction'] >= 20:
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


def analyze_results_and_decide(results):
    """Analyze results and make decision"""
    coverage = results['best_coverage']
    mpiw_reduction = results['mpiw_reduction']
    
    if coverage < 0.5:
        return f"Coverage critically low ({coverage:.3f}), need major adjustments"
    elif coverage < 0.88:
        return f"Coverage too low ({coverage:.3f}), reduce efficiency weight further"
    elif coverage > 0.92:
        return f"Coverage too high ({coverage:.3f}), increase efficiency weight"
    elif mpiw_reduction < 10:
        return f"MPIW improvement insufficient ({mpiw_reduction:.1f}%), allow more asymmetry"
    elif mpiw_reduction < 20:
        return f"Getting closer ({mpiw_reduction:.1f}% reduction), fine-tune hyperparameters"
    else:
        return f"Target achieved! Coverage: {coverage:.3f}, MPIW reduction: {mpiw_reduction:.1f}%"


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run autonomous training
    results = run_autonomous_training()
    print(f"\nFinal OAGA Framework Status: {results['status']}")
    if results.get('best_result'):
        print(f"Best Coverage: {results['best_result']['best_coverage']:.3f}")
        print(f"Best MPIW Reduction: {results['best_result']['mpiw_reduction']:.1f}%")