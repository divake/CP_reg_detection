import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoringMLP(nn.Module):
    """
    MLP architecture for the learnable scoring function.
    
    This network takes features from object detection predictions 
    and outputs learned nonconformity scores for each bounding box coordinate.
    """
    def __init__(self, input_dim=10, hidden_dims=[128, 64], output_dim=4, dropout_rate=0.2):
        """
        Args:
            input_dim: Dimension of input features (box coords, class scores, etc.)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output scores (typically 4 for box coordinates)
            dropout_rate: Dropout probability for regularization
        """
        super(ScoringMLP, self).__init__()
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())  # Ensure positive scores
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
               Features could include:
               - Box coordinates (normalized)
               - Confidence scores
               - Class probabilities
               - Additional extracted features
        
        Returns:
            scores: Tensor of shape [batch_size, output_dim]
                   Learned nonconformity scores for each box coordinate
        """
        return self.network(x)

def extract_features(pred_boxes, pred_scores, pred_classes_one_hot=None, additional_features=None):
    """
    Extract features from prediction data to feed into the scoring MLP.
    
    Args:
        pred_boxes: Predicted bounding boxes [batch_size, 4]
        pred_scores: Prediction confidence scores [batch_size, 1]
        pred_classes_one_hot: One-hot encoded class predictions [batch_size, num_classes]
        additional_features: Any additional features to include
    
    Returns:
        features: Tensor of combined features
    """
    features = [pred_boxes, pred_scores.unsqueeze(1)]
    
    if pred_classes_one_hot is not None:
        features.append(pred_classes_one_hot)
    
    if additional_features is not None:
        features.append(additional_features)
    
    return torch.cat(features, dim=1)

class CoverageLoss(nn.Module):
    """
    Custom loss function to optimize for coverage.
    
    This loss encourages the model to produce prediction intervals
    that contain the true values at the desired coverage rate.
    """
    def __init__(self, target_coverage=0.9, lambda_width=0.5):
        """
        Args:
            target_coverage: Desired coverage rate (e.g., 0.9 for 90%)
            lambda_width: Weight for the interval width penalty
        """
        super(CoverageLoss, self).__init__()
        self.target_coverage = target_coverage
        self.lambda_width = lambda_width
        
    def forward(self, pred_lower, pred_upper, gt_values, tau=None):
        """
        Calculate the coverage loss.
        
        Args:
            pred_lower: Lower bounds of prediction intervals
            pred_upper: Upper bounds of prediction intervals
            gt_values: Ground truth values
            tau: Current quantile threshold (optional)
            
        Returns:
            loss: Combined coverage and efficiency loss
        """
        # Check if values are in the prediction interval
        in_interval = (gt_values >= pred_lower) & (gt_values <= pred_upper)
        
        # Calculate empirical coverage
        coverage = torch.mean(in_interval.float(), dim=0)
        
        # Coverage loss: penalize when coverage is below target
        coverage_penalty = torch.relu(self.target_coverage - coverage).mean()
        
        # Efficiency loss: penalize wide intervals
        width = (pred_upper - pred_lower).mean()
        
        # Combined loss
        loss = coverage_penalty + self.lambda_width * width
        
        return loss 