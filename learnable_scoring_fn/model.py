import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoringMLP(nn.Module):
    """
    MLP architecture for the learnable scoring function.
    
    This network takes features from object detection predictions 
    and outputs a single learned nonconformity score.
    """
    
    def __init__(self, input_dim: int = 13, hidden_dims: list = [128, 64, 32], 
                 output_dim: int = 1, dropout_rate: float = 0.2):
        """
        Args:
            input_dim: Dimension of input features (coordinates + confidence + hand-crafted features)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output scores (1 for single score)
            dropout_rate: Dropout probability for regularization
        """
        super(ScoringMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer - just linear, no activation
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Initialize the final layer to output around 1.0
        final_layer = list(self.modules())[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, 1.0)
            nn.init.normal_(final_layer.weight, mean=0, std=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
               Features include:
               - Box coordinates (x0, y0, x1, y1)
               - Confidence scores
               - Hand-crafted features (area, aspect ratio, etc.)
        
        Returns:
            scores: Tensor of shape [batch_size, 1]
                   Learned nonconformity scores (always positive)
        """
        raw_output = self.network(x)
        # Use exponential to ensure positive outputs, but clamp to prevent extreme values
        output = torch.exp(raw_output)
        return torch.clamp(output, min=0.1, max=10.0)
    
    def get_model_info(self) -> dict:
        """Return model configuration information."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class CoverageLoss(nn.Module):
    """
    Custom loss function that balances coverage and efficiency.
    
    This loss encourages the model to achieve target coverage (90%) 
    while minimizing prediction interval width.
    """
    
    def __init__(self, target_coverage: float = 0.9, lambda_width: float = 0.1):
        """
        Args:
            target_coverage: Target coverage level (e.g., 0.9 for 90%)
            lambda_width: Weight for interval width penalty
        """
        super(CoverageLoss, self).__init__()
        self.target_coverage = target_coverage
        self.lambda_width = lambda_width
    
    def forward(self, pred_intervals_lower: torch.Tensor, 
                pred_intervals_upper: torch.Tensor, 
                gt_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute coverage + efficiency loss.
        
        Args:
            pred_intervals_lower: [batch_size, 4] lower bounds of prediction intervals
            pred_intervals_upper: [batch_size, 4] upper bounds of prediction intervals
            gt_coords: [batch_size, 4] ground truth coordinates
            
        Returns:
            loss: Scalar loss value
        """
        # Check which ground truth points are covered by intervals
        covered = (gt_coords >= pred_intervals_lower) & (gt_coords <= pred_intervals_upper)
        
        # Coverage loss: penalize deviations from target coverage
        actual_coverage = covered.float().mean()
        coverage_loss = (actual_coverage - self.target_coverage) ** 2
        
        # Width loss: penalize wide intervals
        interval_widths = pred_intervals_upper - pred_intervals_lower
        avg_width = interval_widths.mean()
        
        # Normalize width by coordinate magnitudes to make it scale-invariant
        coord_magnitudes = torch.abs(gt_coords).mean()
        normalized_width = avg_width / (coord_magnitudes + 1e-6)
        
        # Total loss
        total_loss = coverage_loss + self.lambda_width * normalized_width
        
        return total_loss


class AdaptiveLambdaScheduler:
    """
    Scheduler for adaptive lambda_width during training.
    
    Implements curriculum learning: start with low lambda (focus on coverage),
    gradually increase to balance coverage and efficiency.
    """
    
    def __init__(self, initial_lambda: float = 0.01, final_lambda: float = 0.1, 
                 warmup_epochs: int = 20, ramp_epochs: int = 30):
        """
        Args:
            initial_lambda: Starting lambda value
            final_lambda: Final lambda value  
            warmup_epochs: Epochs to keep initial lambda
            ramp_epochs: Epochs to ramp from initial to final lambda
        """
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.total_transition_epochs = warmup_epochs + ramp_epochs
    
    def get_lambda(self, epoch: int) -> float:
        """Get lambda value for current epoch."""
        if epoch < self.warmup_epochs:
            # Warmup phase: keep initial lambda
            return self.initial_lambda
        elif epoch < self.total_transition_epochs:
            # Ramp phase: linearly increase lambda
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            return self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
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