"""Loss functions for symmetric adaptive conformal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SymmetricAdaptiveLoss(nn.Module):
    """
    Loss function for symmetric adaptive conformal prediction.
    
    Loss = Coverage Loss + λ * Efficiency Loss
    
    Coverage Loss: Smooth penalty when GT is outside prediction interval
    Efficiency Loss: Mean prediction interval width normalized by object size
    
    Uses smooth losses (not binary) for stable training and proper gradients.
    """
    
    def __init__(
        self,
        target_coverage: float = 0.9,
        lambda_efficiency: float = 0.1,
        coverage_loss_type: str = 'smooth_l1',
        size_normalization: bool = True
    ):
        """
        Initialize the loss function.
        
        Args:
            target_coverage: Target coverage level (e.g., 0.9 for 90%)
            lambda_efficiency: Weight for efficiency loss
            coverage_loss_type: Type of coverage loss ('smooth_l1', 'huber', 'mse')
            size_normalization: Whether to normalize MPIW by object size
        """
        super().__init__()
        
        self.target_coverage = target_coverage
        self.lambda_efficiency = lambda_efficiency
        self.coverage_loss_type = coverage_loss_type
        self.size_normalization = size_normalization
        
        # For smooth L1 and Huber losses
        self.smooth_l1_beta = 1.0
        self.huber_delta = 1.0
        
    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        predicted_widths: torch.Tensor,
        tau: float = 1.0,
        return_components: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the symmetric adaptive loss.
        
        Args:
            pred_boxes: Predicted bounding boxes [batch_size, 4]
            gt_boxes: Ground truth bounding boxes [batch_size, 4]
            predicted_widths: Model's width predictions [batch_size, 4]
            tau: Current calibration factor
            return_components: Whether to return individual loss components
            
        Returns:
            Dictionary containing:
                - 'total': Total loss
                - 'coverage': Coverage loss component
                - 'efficiency': Efficiency loss component
                - 'coverage_rate': Actual coverage rate in batch
                - 'avg_mpiw': Average MPIW
                - 'normalized_mpiw': Size-normalized MPIW
        """
        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device
        
        # Apply tau scaling to get calibrated widths
        scaled_widths = predicted_widths * tau
        
        # Form symmetric intervals
        lower_bounds = pred_boxes - scaled_widths
        upper_bounds = pred_boxes + scaled_widths
        
        # Compute coverage violations
        violations_lower = F.relu(gt_boxes - upper_bounds)  # GT above upper bound
        violations_upper = F.relu(lower_bounds - gt_boxes)  # GT below lower bound
        
        # Total violations per coordinate
        violations = violations_lower + violations_upper  # [batch_size, 4]
        
        # Coverage loss based on selected type
        if self.coverage_loss_type == 'smooth_l1':
            # Smooth L1 loss for violations
            coverage_loss_per_coord = F.smooth_l1_loss(
                violations,
                torch.zeros_like(violations),
                reduction='none',
                beta=self.smooth_l1_beta
            )
        elif self.coverage_loss_type == 'huber':
            # Huber loss for violations
            coverage_loss_per_coord = F.huber_loss(
                violations,
                torch.zeros_like(violations),
                reduction='none',
                delta=self.huber_delta
            )
        else:  # 'mse'
            # Mean squared error for violations
            coverage_loss_per_coord = violations ** 2
        
        # Average across coordinates and batch
        coverage_loss = coverage_loss_per_coord.mean()
        
        # Efficiency loss: MPIW
        # Since intervals are symmetric, total width is 2 * predicted_width
        interval_widths = 2 * scaled_widths  # [batch_size, 4]
        
        # Average width per box (mean across 4 coordinates)
        mpiw_per_box = interval_widths.mean(dim=1)  # [batch_size]
        
        if self.size_normalization:
            # Normalize by object size
            box_widths = gt_boxes[:, 2] - gt_boxes[:, 0]  # x1 - x0
            box_heights = gt_boxes[:, 3] - gt_boxes[:, 1]  # y1 - y0
            
            # Average dimension as size proxy
            object_sizes = (box_widths + box_heights) / 2 + 1.0  # +1 to avoid division by zero
            
            # Normalized MPIW
            normalized_mpiw = mpiw_per_box / object_sizes
            efficiency_loss = normalized_mpiw.mean()
        else:
            # Raw MPIW
            efficiency_loss = mpiw_per_box.mean()
            normalized_mpiw = mpiw_per_box  # For logging
        
        # Total loss
        total_loss = coverage_loss + self.lambda_efficiency * efficiency_loss
        
        # Compute actual coverage rate for monitoring
        with torch.no_grad():
            # Check if GT is within intervals for all coordinates
            covered_lower = gt_boxes >= lower_bounds  # [batch_size, 4]
            covered_upper = gt_boxes <= upper_bounds  # [batch_size, 4]
            covered = covered_lower & covered_upper  # [batch_size, 4]
            
            # Box is covered if ALL coordinates are covered
            box_covered = covered.all(dim=1)  # [batch_size]
            coverage_rate = box_covered.float().mean()
            
            # Additional statistics
            avg_mpiw = mpiw_per_box.mean()
            avg_normalized_mpiw = normalized_mpiw.mean()
            
            # Per-coordinate coverage (for debugging)
            coord_coverage = covered.float().mean(dim=0)
        
        # Return results
        result = {
            'total': total_loss,
            'coverage': coverage_loss,
            'efficiency': efficiency_loss,
            'coverage_rate': coverage_rate,
            'avg_mpiw': avg_mpiw,
            'normalized_mpiw': avg_normalized_mpiw
        }
        
        if return_components:
            # Add detailed statistics
            result.update({
                'coord_coverage': coord_coverage,  # Coverage per coordinate
                'avg_widths': scaled_widths.mean(dim=0),  # Average width per coordinate
                'min_widths': scaled_widths.min(dim=0)[0],
                'max_widths': scaled_widths.max(dim=0)[0]
            })
        
        return result


class SmoothQuantileLoss(nn.Module):
    """
    Alternative loss based on smooth quantile regression.
    
    This can be more stable for learning calibrated uncertainties.
    """
    
    def __init__(self, quantile: float = 0.9):
        super().__init__()
        self.quantile = quantile
        
    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        predicted_widths: torch.Tensor,
        tau: float = 1.0
    ) -> torch.Tensor:
        """
        Compute smooth quantile loss.
        
        For conformal prediction, we want the tau-scaled width to be the
        quantile of the absolute errors.
        """
        # Compute absolute errors
        errors = torch.abs(gt_boxes - pred_boxes)  # [batch_size, 4]
        
        # Scaled widths
        scaled_widths = predicted_widths * tau
        
        # Quantile loss (pinball loss)
        residuals = errors - scaled_widths
        
        quantile_loss = torch.mean(
            torch.max(
                self.quantile * residuals,
                (self.quantile - 1) * residuals
            )
        )
        
        return quantile_loss