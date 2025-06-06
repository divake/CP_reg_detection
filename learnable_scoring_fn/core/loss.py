"""Loss functions for learnable scoring functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class RegressionCoverageLoss(nn.Module):
    """
    Coverage loss for regression-based conformal prediction.
    
    This loss ensures that ground truth falls within the predicted intervals
    while minimizing interval width for efficiency.
    """
    
    def __init__(self, target_coverage: float = 0.9, efficiency_weight: float = 0.1,
                 calibration_weight: float = 0.05):
        """
        Args:
            target_coverage: Target coverage level (e.g., 0.9 for 90%)
            efficiency_weight: Weight for interval width penalty
            calibration_weight: Weight for calibration loss
        """
        super(RegressionCoverageLoss, self).__init__()
        self.target_coverage = target_coverage
        self.efficiency_weight = efficiency_weight
        self.calibration_weight = calibration_weight
    
    def forward(self, widths: torch.Tensor, gt_coords: torch.Tensor, 
                pred_coords: torch.Tensor, tau: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute regression coverage loss with CORRECT interval coverage.
        
        Args:
            widths: [batch_size, 1] predicted interval widths
            gt_coords: [batch_size, 4] ground truth coordinates
            pred_coords: [batch_size, 4] predicted coordinates
            tau: Current tau value (scalar or tensor)
            
        Returns:
            losses: Dictionary containing individual loss components
        """
        batch_size = widths.size(0)
        
        # Calculate actual errors
        errors = torch.abs(gt_coords - pred_coords)  # [batch_size, 4]
        
        # Calculate prediction interval bounds
        interval_half_widths = widths * tau  # [batch_size, 1]
        interval_half_widths_expanded = interval_half_widths.expand(-1, 4)  # [batch_size, 4]
        
        # CORRECT: Form prediction intervals
        lower_bounds = pred_coords - interval_half_widths_expanded
        upper_bounds = pred_coords + interval_half_widths_expanded
        
        # CORRECT: Check if ground truth falls within intervals
        # Coverage = 1 if gt is within [lower, upper], 0 otherwise
        covered_per_coord = (gt_coords >= lower_bounds) & (gt_coords <= upper_bounds)  # [batch_size, 4]
        
        # For bounding boxes: ALL coordinates must be covered
        sample_covered = covered_per_coord.all(dim=1).float()  # [batch_size]
        actual_coverage = sample_covered.mean()
        
        # 1. Coverage Loss - penalize under-coverage more than over-coverage
        coverage_error = actual_coverage - self.target_coverage
        if coverage_error < 0:  # Under-coverage
            coverage_loss = coverage_error ** 2 * 10.0  # Heavily penalize
        else:  # Over-coverage
            coverage_loss = coverage_error ** 2
        
        # 2. Efficiency Loss - directly minimize average interval width
        # No normalization by error - we want absolute efficiency
        efficiency_loss = widths.mean()
        
        # 3. Calibration Loss - encourage proportionality between widths and actual errors
        # Widths should be proportional to the expected error magnitude
        avg_errors_per_sample = errors.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Use correlation-based calibration loss
        # High correlation means widths adapt to error patterns
        error_mean = avg_errors_per_sample.mean()
        width_mean = widths.mean()
        
        error_centered = avg_errors_per_sample - error_mean
        width_centered = widths - width_mean
        
        covariance = (error_centered * width_centered).mean()
        error_std = error_centered.pow(2).mean().sqrt() + 1e-6
        width_std = width_centered.pow(2).mean().sqrt() + 1e-6
        
        correlation = covariance / (error_std * width_std)
        calibration_loss = 1.0 - correlation  # Want high correlation
        
        # Combine losses with adaptive weighting
        if actual_coverage < self.target_coverage - 0.3:  # Way under coverage (< 60%)
            # Heavily prioritize coverage, almost ignore efficiency
            total_loss = coverage_loss + 0.0001 * self.efficiency_weight * efficiency_loss
        elif actual_coverage < self.target_coverage - 0.1:  # Under coverage (< 80%)
            # Prioritize coverage, some efficiency
            total_loss = coverage_loss + 0.01 * self.efficiency_weight * efficiency_loss
        else:
            # Normal weighting
            total_loss = (coverage_loss + 
                         self.efficiency_weight * efficiency_loss +
                         self.calibration_weight * calibration_loss)
        
        # Return detailed losses for monitoring
        losses = {
            'total': total_loss,
            'coverage': coverage_loss,
            'efficiency': efficiency_loss,
            'calibration': calibration_loss,
            'actual_coverage': actual_coverage,
            'avg_width': widths.mean(),
            'correlation': correlation
        }
        
        return losses


def calculate_tau_regression(widths: torch.Tensor, errors: torch.Tensor, 
                            target_coverage: float = 0.9) -> torch.Tensor:
    """
    Calculate tau for regression conformal prediction without circular dependency.
    
    In the fixed approach, we use tau=1.0 and let the model learn appropriate widths.
    This avoids the circular dependency where tau depends on the widths being learned.
    
    Args:
        widths: [n_cal, 1] predicted interval widths from scoring function (not used)
        errors: [n_cal, 4] absolute errors between predictions and ground truth
        target_coverage: Desired coverage level
        
    Returns:
        tau: Fixed value of 1.0
    """
    # Use fixed tau = 1.0
    # The model will learn to output widths that achieve target coverage
    # when multiplied by tau = 1.0
    return torch.tensor(1.0, device=widths.device)