"""Symmetric Adaptive MLP for conformal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class SymmetricAdaptiveMLP(nn.Module):
    """
    Predicts symmetric interval widths for each coordinate.
    
    Output: [wx0, wy0, wx1, wy1] where:
    - wx0: symmetric width for x0 coordinate (left/right)
    - wy0: symmetric width for y0 coordinate (top/bottom)
    - wx1: symmetric width for x1 coordinate (left/right)
    - wy1: symmetric width for y1 coordinate (top/bottom)
    
    The model learns to predict appropriate widths based on:
    - Object size and aspect ratio
    - Position in image
    - Confidence scores
    - Other geometric features
    """
    
    def __init__(
        self, 
        input_dim: int = 17,
        hidden_dims: list = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        """
        Initialize the symmetric MLP.
        
        Args:
            input_dim: Dimension of input features (default: 17)
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
            dropout_rate: Dropout probability for regularization
            activation: Activation function type ('relu', 'elu', 'gelu')
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation_type = activation
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (if enabled)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer - 4 width predictions
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 4)
        
        # Initialize weights
        self._initialize_weights()
        
        # For logging purposes
        self.training_step = 0
        
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/He initialization depending on activation
                if self.activation_type == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                    
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict symmetric interval widths.
        
        Args:
            x: Input tensor of shape [batch_size, 17]
               Features include coordinates, confidence, geometric features,
               and uncertainty indicators
        
        Returns:
            widths: Tensor of shape [batch_size, 4]
                   Predicted symmetric widths for each coordinate
                   All values are positive (enforced by softplus + 1.0)
        """
        # Pass through feature layers
        features = self.feature_layers(x)
        
        # Get raw width predictions
        raw_widths = self.output_layer(features)
        
        # Ensure positive outputs with softplus
        # Add 1.0 to ensure minimum width of 1 pixel
        widths = F.softplus(raw_widths) + 1.0
        
        # Log statistics during training (every 100 steps)
        if self.training and self.training_step % 100 == 0:
            with torch.no_grad():
                mean_widths = widths.mean(dim=0)
                std_widths = widths.std(dim=0)
                print(f"Step {self.training_step} - Width stats:")
                print(f"  Mean: {mean_widths.detach().cpu().numpy()}")
                print(f"  Std:  {std_widths.detach().cpu().numpy()}")
        
        if self.training:
            self.training_step += 1
        
        return widths
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving/loading."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'use_batch_norm': self.use_batch_norm
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create model from configuration dictionary."""
        return cls(**config)
    
    def predict_intervals(
        self, 
        features: torch.Tensor,
        predictions: torch.Tensor,
        tau: float = 1.0
    ) -> tuple:
        """
        Predict symmetric intervals given features and predictions.
        
        Args:
            features: Input features [batch_size, 17]
            predictions: Predicted boxes [batch_size, 4]
            tau: Calibration factor (default: 1.0)
            
        Returns:
            lower_bounds: Lower bounds of intervals [batch_size, 4]
            upper_bounds: Upper bounds of intervals [batch_size, 4]
        """
        # Get width predictions
        widths = self.forward(features)
        
        # Apply tau scaling
        calibrated_widths = widths * tau
        
        # Create symmetric intervals
        lower_bounds = predictions - calibrated_widths
        upper_bounds = predictions + calibrated_widths
        
        return lower_bounds, upper_bounds
    
    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return f"SymmetricMLP_h{'x'.join(map(str, self.hidden_dims))}"