"""
Gated Asymmetry with Stable Parameterization (GAP) Model
Learns to split uncertainty between inner/outer boundaries using a gating mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAsymmetryNetwork(nn.Module):
    def __init__(self, input_dim=17, hidden_dims=[256, 128, 64], dropout=0.1, activation='elu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        
        # Select activation function
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ELU()
        
        # Base uncertainty network: predicts total uncertainty needed
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 4))
        self.base_uncertainty_net = nn.Sequential(*layers)
        
        # Gating network: decides how to split uncertainty
        gate_layers = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims[1:]):  # Smaller network for gates
            gate_layers.extend([
                nn.Linear(prev_dim, dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        gate_layers.append(nn.Linear(prev_dim, 4))
        self.gating_net = nn.Sequential(*gate_layers)
        
        # Learnable tau parameter (initialized to 1.0)
        self.tau = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
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
        
        # Total uncertainty (always positive) - increased base range
        base_widths = F.softplus(self.base_uncertainty_net(features)) * 10.0 + 5.0  # [B, 4]
        
        # Gates determine inner/outer split [0,1]
        gates = torch.sigmoid(self.gating_net(features))  # [B, 4]
        
        # Size modulation based on box area
        if boxes is not None:
            # Calculate box areas
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights
            
            # Size-based modulation factor (similar to baseline)
            size_factor = torch.sqrt(areas / 5000.0).clamp(0.5, 3.0)
            size_factor = size_factor.unsqueeze(1).expand(-1, 4)
            
            # Apply size modulation
            base_widths = base_widths * size_factor
        
        # Stable offset parameterization
        inner_offset = -base_widths * gates * self.tau  # Can shrink
        outer_offset = base_widths * (1 - gates) * self.tau  # Can expand
        
        # Add small noise during training to prevent collapse
        if self.training:
            noise = torch.randn_like(inner_offset) * 0.1
            inner_offset = inner_offset + noise
            outer_offset = outer_offset + noise
        
        return inner_offset, outer_offset
    
    def get_tau(self):
        """Get current tau value"""
        return self.tau.item()
    
    def set_tau(self, tau_value):
        """Set tau value"""
        with torch.no_grad():
            self.tau.copy_(torch.tensor(tau_value))