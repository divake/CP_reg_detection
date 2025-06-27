"""
Loss functions for OAGA (Ordering-Aware Gated Asymmetry)
Includes strong ordering constraints and gradual asymmetry learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/ssd_4TB/divake/conformal-od/learnable_scoring_fn')


class OAGALoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Loss weights
        self.lambda_coverage = config.get('lambda_coverage', 1.0)
        self.lambda_efficiency = config.get('lambda_efficiency', 0.01)  # Start very low
        self.lambda_ordering = config.get('lambda_ordering', 0.5)  # Strong ordering constraint
        self.lambda_asymmetry = config.get('lambda_asymmetry', 0.1)  # Regularize asymmetry
        
        # Size thresholds and targets
        self.size_thresholds = config.get('size_thresholds', {'small': 32**2, 'large': 96**2})
        self.size_targets = config.get('size_targets', {'small': 0.90, 'medium': 0.89, 'large': 0.85})
        
        # Coverage and efficiency weights per size
        self.coverage_weights = config.get('coverage_weights', {'small': 1.5, 'medium': 1.0, 'large': 0.7})
        self.efficiency_weights = config.get('efficiency_weights', {'small': 0.5, 'medium': 1.0, 'large': 2.0})
        
        # Asymmetry warmup
        self.asymmetry_warmup_epochs = config.get('asymmetry_warmup_epochs', 10)
        self.current_epoch = 0
        
    def forward(self, inner_offset, outer_offset, boxes, ground_truth):
        """
        Args:
            inner_offset: [batch_size, 4] negative offsets for shrinking
            outer_offset: [batch_size, 4] positive offsets for expanding
            boxes: [batch_size, 4] predicted boxes
            ground_truth: [batch_size, 4] ground truth boxes
        """
        batch_size = boxes.shape[0]
        device = boxes.device
        
        # Compute inner and outer boxes
        inner_boxes = boxes + inner_offset  # Shrunk boxes
        outer_boxes = boxes + outer_offset  # Expanded boxes
        
        # 1. Ordering Loss - ensure inner is strictly inside outer
        ordering_violations = F.relu(inner_boxes - outer_boxes + 1e-3)  # Small margin
        ordering_loss = ordering_violations.mean()
        
        # 2. Coverage Loss
        # Check if ground truth is within [inner, outer] interval
        coverage_mask = self._check_coverage(inner_boxes, outer_boxes, ground_truth)
        
        # Calculate box areas for size stratification
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # Size stratification
        small_mask = areas < self.size_thresholds['small']
        large_mask = areas > self.size_thresholds['large']
        medium_mask = ~(small_mask | large_mask)
        
        # Size-stratified coverage
        overall_coverage = coverage_mask.float().mean()
        coverage_small = coverage_mask[small_mask].float().mean() if small_mask.any() else torch.tensor(0.9).to(device)
        coverage_medium = coverage_mask[medium_mask].float().mean() if medium_mask.any() else torch.tensor(0.89).to(device)
        coverage_large = coverage_mask[large_mask].float().mean() if large_mask.any() else torch.tensor(0.85).to(device)
        
        # Coverage losses with strong focus on achieving target
        coverage_loss_small = (self.size_targets['small'] - coverage_small).abs()
        coverage_loss_medium = (self.size_targets['medium'] - coverage_medium).abs()
        coverage_loss_large = (self.size_targets['large'] - coverage_large).abs()
        
        # Weighted coverage loss
        coverage_loss = (
            self.coverage_weights['small'] * coverage_loss_small +
            self.coverage_weights['medium'] * coverage_loss_medium +
            self.coverage_weights['large'] * coverage_loss_large
        ) / 3.0
        
        # Add strong penalty if overall coverage is too low
        if overall_coverage < 0.5:
            coverage_loss = coverage_loss + 10.0 * (0.5 - overall_coverage)
        elif overall_coverage < 0.8:
            coverage_loss = coverage_loss + 2.0 * (0.8 - overall_coverage)
        
        # 3. Efficiency Loss (MPIW)
        interval_widths = outer_boxes - inner_boxes  # [B, 4]
        mpiw = interval_widths.mean(dim=1)  # [B]
        
        # Size-normalized efficiency
        normalized_mpiw = mpiw / torch.sqrt(areas + 1e-6)
        
        # Efficiency loss with warmup
        efficiency_weight = self._get_efficiency_weight()
        efficiency_loss = efficiency_weight * normalized_mpiw.mean()
        
        # 4. Asymmetry Regularization
        # Encourage starting symmetric and gradually allowing asymmetry
        total_width = outer_offset - inner_offset
        inner_ratio = -inner_offset / (total_width + 1e-6)
        asymmetry_loss = (inner_ratio - 0.5).abs().mean()
        
        # Apply asymmetry warmup
        asymmetry_weight = self._get_asymmetry_weight()
        asymmetry_loss = asymmetry_weight * asymmetry_loss
        
        # Total loss
        total_loss = (
            self.lambda_coverage * coverage_loss +
            self.lambda_efficiency * efficiency_loss +
            self.lambda_ordering * ordering_loss +
            self.lambda_asymmetry * asymmetry_loss
        )
        
        # Return detailed metrics for logging
        metrics = {
            'total_loss': total_loss,
            'coverage_loss': coverage_loss,
            'efficiency_loss': efficiency_loss,
            'ordering_loss': ordering_loss,
            'asymmetry_loss': asymmetry_loss,
            'ordering_violations': (ordering_violations > 0).float().mean().item(),
            'coverage_small': coverage_small.item() if small_mask.any() else 0.9,
            'coverage_medium': coverage_medium.item() if medium_mask.any() else 0.89,
            'coverage_large': coverage_large.item() if large_mask.any() else 0.85,
            'mpiw_small': mpiw[small_mask].mean().item() if small_mask.any() else 0.0,
            'mpiw_medium': mpiw[medium_mask].mean().item() if medium_mask.any() else 0.0,
            'mpiw_large': mpiw[large_mask].mean().item() if large_mask.any() else 0.0,
            'overall_coverage': overall_coverage.item(),
            'overall_mpiw': mpiw.mean().item(),
            'inner_ratio': inner_ratio.mean().item()
        }
        
        return total_loss, metrics
    
    def _check_coverage(self, inner_boxes, outer_boxes, ground_truth):
        """Check if ground truth is within [inner, outer] interval"""
        # Ground truth should be >= inner and <= outer
        within_inner = (ground_truth >= inner_boxes).all(dim=1)
        within_outer = (ground_truth <= outer_boxes).all(dim=1)
        return within_inner & within_outer
    
    def _get_efficiency_weight(self):
        """Get efficiency weight with warmup"""
        if self.current_epoch < self.asymmetry_warmup_epochs:
            # Start with very low efficiency weight
            return 0.1
        else:
            # Gradually increase
            return 1.0
    
    def _get_asymmetry_weight(self):
        """Get asymmetry regularization weight with warmup"""
        if self.current_epoch < self.asymmetry_warmup_epochs:
            # Strong regularization initially
            return 1.0
        else:
            # Gradually decrease to allow more asymmetry
            progress = (self.current_epoch - self.asymmetry_warmup_epochs) / 10
            return max(0.1, 1.0 - progress)
    
    def set_epoch(self, epoch):
        """Update current epoch for warmup scheduling"""
        self.current_epoch = epoch