"""
Loss functions for GAP (Gated Asymmetry with Stable Parameterization)
Adapts the baseline size-aware loss for asymmetric predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/ssd_4TB/divake/conformal-od/learnable_scoring_fn')

# Import baseline losses
from core_symmetric.losses.size_aware_loss import SizeAwareSymmetricLoss


class GAPLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_efficiency = config.get('lambda_efficiency', 0.25)
        self.lambda_coverage = config.get('lambda_coverage', 1.0)
        self.size_thresholds = config.get('size_thresholds', {'small': 32**2, 'large': 96**2})
        self.size_targets = config.get('size_targets', {'small': 0.90, 'medium': 0.89, 'large': 0.85})
        
        # Coverage and efficiency weights per size
        self.coverage_weights = config.get('coverage_weights', {'small': 1.5, 'medium': 1.0, 'large': 0.7})
        self.efficiency_weights = config.get('efficiency_weights', {'small': 0.5, 'medium': 1.0, 'large': 2.0})
        
    def forward(self, inner_offset, outer_offset, boxes, ground_truth):
        """
        Args:
            inner_offset: [batch_size, 4] negative offsets for shrinking
            outer_offset: [batch_size, 4] positive offsets for expanding
            boxes: [batch_size, 4] predicted boxes
            ground_truth: [batch_size, 4] ground truth boxes
        """
        batch_size = boxes.shape[0]
        
        # Compute inner and outer boxes
        inner_boxes = boxes + inner_offset  # Shrunk boxes
        outer_boxes = boxes + outer_offset  # Expanded boxes
        
        # Ensure ordering: inner should be inside outer
        ordering_violation = F.relu(inner_boxes - outer_boxes).mean()
        
        # Calculate box areas for size stratification
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # Size stratification
        small_mask = areas < self.size_thresholds['small']
        large_mask = areas > self.size_thresholds['large']
        medium_mask = ~(small_mask | large_mask)
        
        # Coverage calculation
        coverage_mask = self._check_coverage(inner_boxes, outer_boxes, ground_truth)
        
        # Size-stratified coverage
        coverage_small = coverage_mask[small_mask].float().mean() if small_mask.any() else torch.tensor(0.0)
        coverage_medium = coverage_mask[medium_mask].float().mean() if medium_mask.any() else torch.tensor(0.0)
        coverage_large = coverage_mask[large_mask].float().mean() if large_mask.any() else torch.tensor(0.0)
        
        # Coverage losses
        coverage_loss_small = F.smooth_l1_loss(coverage_small, torch.tensor(self.size_targets['small']).to(coverage_small.device))
        coverage_loss_medium = F.smooth_l1_loss(coverage_medium, torch.tensor(self.size_targets['medium']).to(coverage_medium.device))
        coverage_loss_large = F.smooth_l1_loss(coverage_large, torch.tensor(self.size_targets['large']).to(coverage_large.device))
        
        # Weighted coverage loss
        coverage_loss = (
            self.coverage_weights['small'] * coverage_loss_small +
            self.coverage_weights['medium'] * coverage_loss_medium +
            self.coverage_weights['large'] * coverage_loss_large
        ) / 3.0
        
        # Efficiency calculation (MPIW)
        interval_widths = outer_boxes - inner_boxes  # [B, 4]
        mpiw = interval_widths.mean(dim=1)  # [B]
        
        # Size-normalized efficiency
        normalized_mpiw_small = (mpiw[small_mask] / torch.sqrt(areas[small_mask])).mean() if small_mask.any() else torch.tensor(0.0)
        normalized_mpiw_medium = (mpiw[medium_mask] / torch.sqrt(areas[medium_mask])).mean() if medium_mask.any() else torch.tensor(0.0)
        normalized_mpiw_large = (mpiw[large_mask] / torch.sqrt(areas[large_mask])).mean() if large_mask.any() else torch.tensor(0.0)
        
        # Weighted efficiency loss
        efficiency_loss = (
            self.efficiency_weights['small'] * normalized_mpiw_small +
            self.efficiency_weights['medium'] * normalized_mpiw_medium +
            self.efficiency_weights['large'] * normalized_mpiw_large
        ) / 3.0
        
        # Coverage penalties - much stronger when coverage is low
        coverage_penalty = 0.0
        overall_coverage = coverage_mask.float().mean()
        
        # Strong penalty when overall coverage is too low
        if overall_coverage < 0.8:
            coverage_penalty += 10.0 * (0.8 - overall_coverage)  # Very strong penalty
        elif overall_coverage < 0.88:
            coverage_penalty += 2.0 * (0.88 - overall_coverage)
            
        # Size-specific penalties
        if coverage_small < 0.88:
            coverage_penalty += 0.5 * (0.88 - coverage_small)
        if coverage_medium < 0.87 or coverage_medium > 0.91:
            coverage_penalty += 0.3 * (F.relu(0.87 - coverage_medium) + F.relu(coverage_medium - 0.91))
        if coverage_large > 0.87:
            coverage_penalty += 0.3 * (coverage_large - 0.87)
        
        # Total loss
        total_loss = (
            self.lambda_coverage * coverage_loss +
            self.lambda_efficiency * efficiency_loss +
            0.1 * ordering_violation +
            coverage_penalty
        )
        
        # Return detailed metrics for logging
        metrics = {
            'total_loss': total_loss,
            'coverage_loss': coverage_loss,
            'efficiency_loss': efficiency_loss,
            'ordering_violation': ordering_violation,
            'coverage_penalty': coverage_penalty,
            'coverage_small': coverage_small.item() if small_mask.any() else 0.0,
            'coverage_medium': coverage_medium.item() if medium_mask.any() else 0.0,
            'coverage_large': coverage_large.item() if large_mask.any() else 0.0,
            'mpiw_small': mpiw[small_mask].mean().item() if small_mask.any() else 0.0,
            'mpiw_medium': mpiw[medium_mask].mean().item() if medium_mask.any() else 0.0,
            'mpiw_large': mpiw[large_mask].mean().item() if large_mask.any() else 0.0,
            'overall_coverage': coverage_mask.float().mean().item(),
            'overall_mpiw': mpiw.mean().item()
        }
        
        return total_loss, metrics
    
    def _check_coverage(self, inner_boxes, outer_boxes, ground_truth):
        """Check if ground truth is within [inner, outer] interval"""
        # Ground truth should be >= inner and <= outer
        within_inner = (ground_truth >= inner_boxes).all(dim=1)
        within_outer = (ground_truth <= outer_boxes).all(dim=1)
        return within_inner & within_outer