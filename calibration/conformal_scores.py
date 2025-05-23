import torch


def abs_res(gt: torch.Tensor, pred: torch.Tensor):
    # Fixed-width absolute residual scores
    return torch.abs(gt - pred)


def norm_res(gt: torch.Tensor, pred: torch.Tensor, unc: torch.Tensor):
    # Scalable normalized residual scores
    return torch.abs(gt - pred) / unc


def quant_res(gt: torch.Tensor, pred_lower: torch.Tensor, pred_upper: torch.Tensor):
    # Scalable CQR scores, see Eq. 6 in the paper
    return torch.max(pred_lower - gt, gt - pred_upper)


def one_sided_res(gt: torch.Tensor, pred: torch.Tensor, min: bool):
    # Fixed-width one-sided scores from Andeol et al. (2023), see Eq. 6 in the paper
    return (pred - gt) if min else (gt - pred)


def one_sided_mult_res(gt: torch.Tensor, pred: torch.Tensor, mult: torch.Tensor, min: bool):
    # Scalable one-sided scores from Andeol et al. (2023), see Eq. 7 in the paper
    return (pred - gt) / mult if min else (gt - pred) / mult


# ===== LEARNABLE SCORING FUNCTION =====

# Global cache for trained model
_trained_scoring_model = None
_trained_feature_extractor = None
_model_path = None


def load_trained_scoring_model(model_path: str = None):
    """
    Load the trained scoring function model and feature extractor.
    
    Args:
        model_path: Path to trained model checkpoint
        
    Returns:
        model: Loaded ScoringMLP model
        feature_extractor: Loaded FeatureExtractor
    """
    global _trained_scoring_model, _trained_feature_extractor, _model_path
    
    # Use default path if not specified
    if model_path is None:
        model_path = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/trained_models/best_model.pt"
    
    # Load model if not cached or path changed
    if (_trained_scoring_model is None or 
        _trained_feature_extractor is None or 
        _model_path != model_path):
        
        try:
            # Import here to avoid circular dependencies
            from learnable_scoring_fn.model import load_model
            from learnable_scoring_fn.feature_utils import FeatureExtractor
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model checkpoint
            model, optimizer, checkpoint = load_model(model_path, device)
            
            # Create and load feature extractor
            feature_extractor = FeatureExtractor()
            if 'feature_stats' in checkpoint and checkpoint['feature_stats'] is not None:
                feature_extractor.feature_stats = checkpoint['feature_stats']
            else:
                # Try to load from separate feature stats file
                import os
                feature_stats_path = os.path.join(os.path.dirname(model_path), 'feature_stats.pt')
                if os.path.exists(feature_stats_path):
                    feature_extractor.load_stats(feature_stats_path)
                else:
                    raise FileNotFoundError(f"Feature stats not found in checkpoint or at {feature_stats_path}")
            
            # Cache the loaded model and extractor
            _trained_scoring_model = model
            _trained_feature_extractor = feature_extractor
            _model_path = model_path
            
            print(f"Loaded trained scoring model from {model_path}")
            
        except Exception as e:
            print(f"Error loading trained scoring model: {e}")
            print("Falling back to absolute residual scoring...")
            return None, None
    
    return _trained_scoring_model, _trained_feature_extractor


def learned_score(gt: torch.Tensor, pred: torch.Tensor, pred_score: torch.Tensor = None, 
                 model_path: str = None, fallback_to_abs: bool = False):
    """
    Learnable scoring function - uses trained model to compute nonconformity scores.
    
    This function acts like abs_res() but uses a trained neural network to compute
    more adaptive nonconformity scores based on prediction features.
    
    Args:
        gt: Ground truth coordinates [N] or [N, 4]
        pred: Predicted coordinates [N] or [N, 4] 
        pred_score: Prediction confidence scores [N] (required for learned scoring)
        model_path: Path to trained model (optional, uses default if None)
        fallback_to_abs: Deprecated - no longer used
        
    Returns:
        scores: Learned nonconformity scores [N] or [N, 4]
    """
    # Handle missing prediction scores
    if pred_score is None:
        raise ValueError("pred_score is required for learned scoring function")
    
    # Load trained model
    model, feature_extractor = load_trained_scoring_model(model_path)
    
    if model is None or feature_extractor is None:
        raise RuntimeError("Failed to load trained scoring model")
    
    device = next(model.parameters()).device
    
    # Ensure tensors are properly shaped
    if pred.dim() == 1:
        # Single coordinate case - reshape to [N, 4] where N=1
        if len(pred) == 1:
            # Single scalar coordinate - create [1, 4] by repeating
            pred_coords = pred.repeat(4).unsqueeze(0)  # [1, 4]
            pred_scores = pred_score.unsqueeze(0) if pred_score.dim() == 0 else pred_score[:1]
            single_coord = True
        else:
            # Multiple coordinates in 1D - reshape as [N, 1] then expand
            n_coords = len(pred)
            pred_coords = pred.unsqueeze(1).expand(-1, 4)  # [N, 4]
            pred_scores = pred_score[:n_coords] if len(pred_score) >= n_coords else pred_score.repeat(n_coords)
            single_coord = False
    elif pred.dim() == 2 and pred.shape[1] == 4:
        # Multi-coordinate case [N, 4] - already correct shape
        pred_coords = pred
        pred_scores = pred_score
        single_coord = False
    else:
        raise ValueError(f"Unexpected pred shape: {pred.shape}. Expected 1D tensor or [N, 4] tensor")
    
    # Move to device
    pred_coords = pred_coords.to(device)
    pred_scores = pred_scores.to(device)
    
    # Extract features
    features = feature_extractor.extract_features(pred_coords, pred_scores)
    
    # Normalize features
    normalized_features = feature_extractor.normalize_features(features)
    
    # Get learned scores
    with torch.no_grad():
        learned_scores = model(normalized_features).squeeze()  # [N]
    
    # Handle different return cases
    if single_coord:
        # Return single score
        return learned_scores.item() if learned_scores.dim() == 0 else learned_scores[0].item()
    else:
        # Return scores for all samples - compute absolute residual and scale by learned score
        gt_tensor = gt.to(device)
        if gt_tensor.dim() == 1:
            if len(gt_tensor) == 1:
                gt_tensor = gt_tensor.repeat(4).unsqueeze(0)  # [1, 4]
            else:
                gt_tensor = gt_tensor.unsqueeze(1).expand(-1, 4)  # [N, 4]
        
        # Compute absolute errors
        abs_errors = torch.abs(gt_tensor - pred_coords)  # [N, 4]
        
        # Scale by learned scores (broadcast learned scores to match coordinates)
        if learned_scores.dim() == 0:
            learned_scores = learned_scores.unsqueeze(0)
        scaled_scores = abs_errors / (learned_scores.unsqueeze(1).expand(-1, 4) + 1e-6)
        
        # Return mean across coordinates for each sample
        return scaled_scores.mean(dim=1).cpu()


def get_learned_score_batch(gt_batch: torch.Tensor, pred_batch: torch.Tensor, 
                           pred_score_batch: torch.Tensor, model_path: str = None):
    """
    Batch version of learned_score for efficiency.
    
    Args:
        gt_batch: Ground truth coordinates [N, 4]
        pred_batch: Predicted coordinates [N, 4]
        pred_score_batch: Prediction confidence scores [N]
        model_path: Path to trained model
        
    Returns:
        scores_batch: Learned nonconformity scores [N]
    """
    # Load model once for the entire batch
    model, feature_extractor = load_trained_scoring_model(model_path)
    
    if model is None or feature_extractor is None:
        raise RuntimeError("Failed to load trained scoring model for batch processing")
    
    device = next(model.parameters()).device
    
    # Move to device
    pred_batch = pred_batch.to(device)
    pred_score_batch = pred_score_batch.to(device)
    gt_batch = gt_batch.to(device)
    
    # Extract and normalize features
    features = feature_extractor.extract_features(pred_batch, pred_score_batch)
    normalized_features = feature_extractor.normalize_features(features)
    
    # Get learned scores
    with torch.no_grad():
        learned_scores = model(normalized_features).squeeze()  # [N]
    
    # Compute scaled nonconformity scores
    abs_errors = torch.abs(gt_batch - pred_batch)  # [N, 4]
    scaled_scores = abs_errors / (learned_scores.unsqueeze(1).expand(-1, 4) + 1e-6)
    
    # Return mean across coordinates
    return scaled_scores.mean(dim=1).cpu()


# Convenience function for integration with existing code
def get_available_scoring_functions():
    """Return list of available scoring function names."""
    return [
        'abs_res',           # Standard absolute residual
        'norm_res',          # Normalized residual (requires uncertainty)
        'quant_res',         # CQR scores (requires prediction intervals)
        'one_sided_res',     # One-sided residual
        'one_sided_mult_res', # One-sided multiplicative residual
        'learned_score'      # Learnable scoring function
    ]


def is_learned_model_available(model_path: str = None):
    """Check if trained model is available and loadable."""
    model, feature_extractor = load_trained_scoring_model(model_path)
    if model is None or feature_extractor is None:
        raise RuntimeError(f"Trained scoring model not available at path: {model_path}")
    return True
