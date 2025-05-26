import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoringMLP(nn.Module):
    """
    MLP architecture for the learnable scoring function following classification framework.
    
    This network takes features from object detection predictions 
    and outputs a single learned nonconformity score following the
    sophisticated approach from the reference classification implementation.
    """
    
    def __init__(self, input_dim: int = 13, hidden_dims: list = [128, 64, 32], 
                 output_dim: int = 1, dropout_rate: float = 0.2, config: dict = None):
        """
        Args:
            input_dim: Dimension of input features (coordinates + confidence + hand-crafted features)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output scores (1 for single score)
            dropout_rate: Dropout probability for regularization
            config: Configuration dictionary for advanced settings
        """
        super(ScoringMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Initialize weights function for conservative initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.5)
        
        # Build network layers with batch normalization
        layers = []
        prev_dim = input_dim
        
        # First layer with special initialization
        first_layer = nn.Linear(prev_dim, hidden_dims[0])
        nn.init.normal_(first_layer.weight, mean=0.0, std=0.05)
        nn.init.constant_(first_layer.bias, 0.5)
        
        layers.append(first_layer)
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        
        # Hidden layers with dropout and batch norm
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout_rate)
            ])
        
        # Final layer with careful initialization
        final_layer = nn.Linear(hidden_dims[-1], 1)
        nn.init.uniform_(final_layer.weight, -0.001, 0.001)
        nn.init.constant_(final_layer.bias, 0.7)
        layers.append(final_layer)
        
        # Final activation to ensure positive outputs
        layers.append(nn.Softplus(beta=1.0))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize remaining layers (skip first layer which we set manually)
        for m in self.network[3:]:
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                init_weights(m)
        
        # Regularization parameters
        self.l2_lambda = 0.001
        self.stability_factor = 0.1
        self.separation_factor = 1.0
        
        # Store current tau for evaluation time separation
        self.tau = 0.3
        
        # Initialize loss components
        self.separation_loss = 0.0
        self.stability_loss = 0.0
        self.l2_reg = 0.0
        
    def set_tau(self, tau):
        """Set current tau value for evaluation-time scoring"""
        self.tau = tau
        
    def forward(self, x: torch.Tensor, true_coverage_target: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the network following classification framework.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
               Features include coordinates, confidence, and hand-crafted features
            true_coverage_target: Optional tensor indicating if this sample should be covered
        
        Returns:
            scores: Tensor of shape [batch_size, 1]
                   Learned nonconformity scores (always positive)
        """
        batch_size = x.size(0)
        
        # Forward pass through network
        scores = self.network(x)
        
        # Ensure scores have proper shape [batch_size, 1]
        if scores.dim() == 1:
            scores = scores.view(batch_size, 1)
        elif scores.shape[1] > 1:
            scores = scores.mean(dim=1, keepdim=True)
        
        # Apply sophisticated regularization during training
        if self.training and true_coverage_target is not None:
            self._compute_training_losses(scores, true_coverage_target)
        else:
            self.separation_loss = 0.0
            self.stability_loss = 0.0
        
        # Apply clamping to keep scores in reasonable range
        scores = torch.clamp(scores, min=0.1, max=10.0)
        
        # Compute L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        return scores
    
    def _compute_training_losses(self, scores: torch.Tensor, true_coverage_target: torch.Tensor):
        """
        Compute sophisticated training losses following classification framework.
        
        Args:
            scores: Current nonconformity scores [batch_size, 1]
            true_coverage_target: Binary tensor [batch_size] indicating if sample should be covered
        """
        scores_squeezed = scores.squeeze(1)  # [batch_size]
        
        # Store scores for loss calculation
        self.true_class_scores = scores_squeezed
        
        # Separation loss: push scores of samples that should be covered toward lower values
        covered_mask = (true_coverage_target == 1).float()
        uncovered_mask = (true_coverage_target == 0).float()
        
        # Gentle separation loss
        covered_score_term = torch.mean(covered_mask * (scores_squeezed ** 2))
        uncovered_score_term = torch.mean(uncovered_mask * torch.relu(2.0 - scores_squeezed))
        
        # Below-tau term for covered samples
        below_tau_term = torch.mean(covered_mask * torch.relu(scores_squeezed - (self.tau * 0.8)) ** 2)
        
        # Combined separation loss
        self.separation_loss = self.separation_factor * (
            0.3 * covered_score_term + 
            0.3 * uncovered_score_term + 
            0.4 * below_tau_term
        )
        
        # Stability loss: encourage consistent outputs for similar inputs
        if self.training:
            perturbed_x = scores.detach() + torch.randn_like(scores) * 0.01
            perturbed_scores = torch.clamp(perturbed_x, min=0.1, max=10.0)
            self.stability_loss = self.stability_factor * torch.mean((scores - perturbed_scores)**2)
        else:
            self.stability_loss = 0.0
    
    def get_total_loss(self):
        """Get total loss including all regularization terms."""
        return self.separation_loss + self.stability_loss + self.l2_reg
    
    def get_model_info(self) -> dict:
        """Return model configuration information."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'l2_lambda': self.l2_lambda,
            'stability_factor': self.stability_factor,
            'separation_factor': self.separation_factor
        }


class CoverageLoss(nn.Module):
    """
    Sophisticated coverage loss following classification framework.
    
    Combines coverage deviation, size penalty, and margin-based separation
    to achieve both high coverage and small prediction sets.
    """
    
    def __init__(self, target_coverage: float = 0.9, lambda_width: float = 0.1, 
                 margin_weight: float = 0.1):
        """
        Args:
            target_coverage: Target coverage level (0.9 for 90% coverage)
            lambda_width: Weight for prediction set size penalty
            margin_weight: Weight for margin-based separation loss
        """
        super(CoverageLoss, self).__init__()
        self.target_coverage = target_coverage
        self.lambda_width = lambda_width
        self.margin_weight = margin_weight
    
    def forward(self, scores: torch.Tensor, tau: torch.Tensor, 
                gt_coords: torch.Tensor = None, pred_coords: torch.Tensor = None) -> torch.Tensor:
        """
        Compute sophisticated coverage loss.
        
        Args:
            scores: Nonconformity scores [batch_size, 1]
            tau: Current tau threshold [1] or [4] for coordinate-wise
            gt_coords: Ground truth coordinates [batch_size, 4] 
            pred_coords: Predicted coordinates [batch_size, 4]
        
        Returns:
            loss: Total coverage loss
        """
        batch_size = scores.size(0)
        scores_squeezed = scores.squeeze(1)  # [batch_size]
        
        # Ensure tau is broadcastable
        if tau.dim() == 0:
            tau_broadcast = tau.expand(batch_size)
        elif tau.size(0) == 1:
            tau_broadcast = tau.expand(batch_size)
        else:
            tau_broadcast = tau[:batch_size] if tau.size(0) >= batch_size else tau.expand(batch_size)
        
        # Coverage calculation: score <= tau means covered
        covered = (scores_squeezed <= tau_broadcast).float()
        current_coverage = covered.mean()
        
        # Coverage deviation loss
        coverage_loss = torch.abs(current_coverage - self.target_coverage)
        
        # Size penalty: encourage smaller prediction sets (higher scores)
        size_penalty = torch.mean(torch.relu(tau_broadcast - scores_squeezed))
        
        # Margin-based loss if ground truth is available
        margin_loss = 0.0
        if gt_coords is not None and pred_coords is not None:
            # Compute coordinate-wise errors
            coord_errors = torch.abs(gt_coords - pred_coords)  # [batch_size, 4]
            avg_error = coord_errors.mean(dim=1)  # [batch_size]
            
            # Encourage higher scores for higher errors
            margin_loss = torch.mean(torch.relu(avg_error - scores_squeezed + 1.0))
        
        # Combine losses
        total_loss = (coverage_loss + 
                     self.lambda_width * size_penalty + 
                     self.margin_weight * margin_loss)
        
        return total_loss


class AdaptiveLambdaScheduler:
    """
    Enhanced scheduler for adaptive lambda_width during training.
    
    Implements sophisticated curriculum learning with multiple phases:
    1. Warmup: focus on coverage
    2. Ramp: gradually balance coverage and efficiency  
    3. Fine-tune: maintain optimal balance
    """
    
    def __init__(self, initial_lambda: float = 0.01, final_lambda: float = 0.1, 
                 warmup_epochs: int = 20, ramp_epochs: int = 30, schedule_type: str = 'linear'):
        """
        Args:
            initial_lambda: Starting lambda value
            final_lambda: Final lambda value  
            warmup_epochs: Epochs to keep initial lambda
            ramp_epochs: Epochs to ramp from initial to final lambda
            schedule_type: Type of schedule ('linear', 'cosine', 'exponential')
        """
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.total_transition_epochs = warmup_epochs + ramp_epochs
        self.schedule_type = schedule_type
    
    def get_lambda(self, epoch: int) -> float:
        """Get lambda value for current epoch with sophisticated scheduling."""
        if epoch < self.warmup_epochs:
            # Warmup phase: keep initial lambda
            return self.initial_lambda
        elif epoch < self.total_transition_epochs:
            # Ramp phase: use specified schedule
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            
            if self.schedule_type == 'linear':
                lambda_val = self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
            elif self.schedule_type == 'cosine':
                lambda_val = self.initial_lambda + 0.5 * (self.final_lambda - self.initial_lambda) * (1 - torch.cos(torch.tensor(progress * 3.14159)).item())
            elif self.schedule_type == 'exponential':
                lambda_val = self.initial_lambda * (self.final_lambda / self.initial_lambda) ** progress
            else:
                lambda_val = self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
                
            return lambda_val
        else:
            # Final phase: keep final lambda
            return self.final_lambda


def save_model(model: ScoringMLP, optimizer: torch.optim.Optimizer, 
               epoch: int, loss: float, model_config: dict, 
               filepath: str, feature_stats: dict = None):
    """
    Save model checkpoint with all necessary information.
    
    Args:
        model: Trained ScoringMLP model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        model_config: Model configuration dictionary
        filepath: Path to save checkpoint
        feature_stats: Feature normalization statistics
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model_config,
        'feature_stats': feature_stats
    }
    torch.save(checkpoint, filepath)


def load_model(filepath: str, device: torch.device = None) -> tuple:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded ScoringMLP model
        optimizer: Loaded optimizer (if available)
        checkpoint: Full checkpoint dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate model from config
    model_config = checkpoint['model_config']
    model = ScoringMLP(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        output_dim=model_config['output_dim'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create optimizer (for potential continued training)
    optimizer = torch.optim.Adam(model.parameters())
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint 