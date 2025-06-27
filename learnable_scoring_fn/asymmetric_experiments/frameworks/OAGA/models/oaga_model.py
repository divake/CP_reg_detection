"""
Ordering-Aware Gated Asymmetry (OAGA) Model
Builds on GAP with ordering constraints and better initialization
Key innovation: Starts from symmetric baseline and learns asymmetry gradually
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OrderingAwareGatedAsymmetry(nn.Module):
    def __init__(self, input_dim=17, hidden_dims=[256, 128, 64], dropout=0.1, 
                 activation='elu', symmetric_init=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.symmetric_init = symmetric_init
        
        # Select activation function
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ELU()
        
        # Symmetric width predictor (like baseline)
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 4))
        self.symmetric_net = nn.Sequential(*layers)
        
        # Asymmetry predictor (learns how to split symmetric width)
        asym_layers = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            asym_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        asym_layers.append(nn.Linear(prev_dim, 4))
        self.asymmetry_net = nn.Sequential(*asym_layers)
        
        # Learnable tau parameter (initialized to match baseline)
        self.tau = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for asymmetry to start symmetric
        if symmetric_init:
            # Initialize asymmetry network to output 0.5 (equal split)
            with torch.no_grad():
                # Set final layer bias to 0 and small weights
                final_layer = self.asymmetry_net[-1]
                final_layer.weight.data *= 0.01
                final_layer.bias.data.fill_(0.0)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, features, boxes=None):
        """
        Args:
            features: [batch_size, 17] feature tensor
            boxes: [batch_size, 4] bounding boxes (x1, y1, x2, y2)
        
        Returns:
            inner_offset: [batch_size, 4] offsets to shrink box (negative values)
            outer_offset: [batch_size, 4] offsets to expand box (positive values)
        """
        batch_size = features.shape[0]
        
        # Predict symmetric width (always positive)
        # Using similar range as baseline model
        symmetric_raw = self.symmetric_net(features)
        symmetric_widths = torch.sigmoid(symmetric_raw) * 30.0 + 3.0  # [3, 33] range
        
        # Size modulation based on box area (like baseline)
        if boxes is not None:
            # Calculate box areas
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights
            
            # Size-based modulation factor
            size_factor = torch.sqrt(areas / 5000.0).clamp(0.5, 3.0)
            size_factor = size_factor.unsqueeze(1).expand(-1, 4)
            
            # Apply size modulation
            symmetric_widths = symmetric_widths * size_factor
        
        # Predict asymmetry split (0 = all inner, 0.5 = symmetric, 1 = all outer)
        asymmetry_raw = self.asymmetry_net(features)
        asymmetry_split = torch.sigmoid(asymmetry_raw)
        
        # Apply soft constraints to prevent extreme asymmetry initially
        # This helps with training stability
        asymmetry_split = 0.2 + 0.6 * asymmetry_split  # Range [0.2, 0.8]
        
        # Compute inner and outer offsets
        # Total width is preserved: |inner| + outer = symmetric_width
        inner_offset = -symmetric_widths * asymmetry_split * self.tau
        outer_offset = symmetric_widths * (1 - asymmetry_split) * self.tau
        
        # Add small noise during training to prevent collapse
        if self.training:
            noise = torch.randn_like(inner_offset) * 0.1
            inner_offset = inner_offset + noise
            outer_offset = outer_offset + noise
        
        # Ensure ordering constraints are softly enforced
        # This helps gradients flow better
        min_gap = 1.0  # Minimum gap between inner and outer
        total_width = outer_offset - inner_offset
        width_constraint = F.relu(min_gap - total_width)
        
        # Apply soft penalty to maintain minimum width
        if width_constraint.mean() > 0:
            # Adjust offsets to maintain minimum gap
            adjustment = width_constraint / 2
            inner_offset = inner_offset - adjustment
            outer_offset = outer_offset + adjustment
        
        return inner_offset, outer_offset
    
    def get_tau(self):
        """Get current tau value"""
        return self.tau.item()
    
    def set_tau(self, tau_value):
        """Set tau value"""
        with torch.no_grad():
            self.tau.copy_(torch.tensor(tau_value))
    
    def get_asymmetry_stats(self):
        """Get statistics about asymmetry predictions"""
        # This is useful for monitoring training
        return {
            'tau': self.get_tau(),
            'symmetric_init': self.symmetric_init
        }