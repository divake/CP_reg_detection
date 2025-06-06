"""Abstract base class for all scoring function models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class BaseScoringFunction(nn.Module, ABC):
    """Abstract base class for all scoring function models.
    
    All scoring functions must:
    1. Accept input features of shape [batch_size, input_dim]
    2. Output positive interval widths of shape [batch_size, 1]
    3. Implement get_config() for saving/loading
    4. Provide a model_name property for logging
    """
    
    def __init__(self, input_dim: int = 17):
        """Initialize base scoring function.
        
        Args:
            input_dim: Dimension of input features (default: 13 geometric + 4 uncertainty)
        """
        super().__init__()
        self.input_dim = input_dim
        self.tau = None  # Stored tau value for inference
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict interval width.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            widths: Positive interval widths [batch_size, 1]
        """
        pass
    
    def ensure_positive_output(self, raw_output: torch.Tensor) -> torch.Tensor:
        """Common method to ensure positive outputs.
        
        This matches the existing MLP behavior to ensure consistency.
        
        Args:
            raw_output: Raw network output [batch_size, 1]
            
        Returns:
            widths: Positive clamped widths [batch_size, 1]
        """
        # Ensure positive width using softplus
        # Initialize to produce widths around 30-35 pixels for higher coverage
        widths = F.softplus(raw_output + 3.5) + 25.0  # Start around ~35 pixels
        
        # Clamp to reasonable range for bounding box coordinates
        widths = torch.clamp(widths, min=5.0, max=100.0)
        
        return widths
    
    def set_tau(self, tau: torch.Tensor):
        """Store tau value for inference.
        
        Args:
            tau: Quantile value from calibration
        """
        self.tau = tau
    
    def get_prediction_intervals(self, predictions: torch.Tensor, widths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate prediction intervals given predictions and learned widths.
        
        Args:
            predictions: [batch_size, 4] predicted bounding box coordinates
            widths: [batch_size, 1] learned interval widths from this model
            
        Returns:
            lower_bounds: [batch_size, 4] lower bounds of prediction intervals
            upper_bounds: [batch_size, 4] upper bounds of prediction intervals
        """
        if self.tau is None:
            raise ValueError("Must set tau using set_tau() before computing intervals")
        
        # Expand widths to match coordinate dimensions
        interval_widths = widths * self.tau  # [batch_size, 1]
        interval_widths = interval_widths.expand(-1, 4)  # [batch_size, 4]
        
        # Calculate intervals
        lower_bounds = predictions - interval_widths
        upper_bounds = predictions + interval_widths
        
        return lower_bounds, upper_bounds
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving.
        
        Returns:
            config: Dictionary containing all hyperparameters needed to recreate the model
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name for logging.
        
        Returns:
            name: Human-readable model name
        """
        pass
    
    def extra_repr(self) -> str:
        """Extra representation for printing model."""
        return f'input_dim={self.input_dim}'