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
                 model_path: str = None, fallback_to_abs: bool = True):
    """
    Learnable scoring function - uses trained model to compute nonconformity scores.
    
    This function acts like abs_res() but uses a trained neural network to compute
    more adaptive nonconformity scores based on prediction features.
    
    Args:
        gt: Ground truth coordinates [N] or [N, 4]
        pred: Predicted coordinates [N] or [N, 4] 
        pred_score: Prediction confidence scores [N] (required for learned scoring)
        model_path: Path to trained model (optional, uses default if None)
        fallback_to_abs: If True, fall back to abs_res if model loading fails
        
    Returns:
        scores: Learned nonconformity scores [N] or [N, 4]
    """
    # Handle missing prediction scores
    if pred_score is None:
        if fallback_to_abs:
            print("Warning: pred_score not provided for learned_score, falling back to abs_res")
            return abs_res(gt, pred)
        else:
            raise ValueError("pred_score is required for learned scoring function")
    
    # Load trained model
    model, feature_extractor = load_trained_scoring_model(model_path)
    
    if model is None or feature_extractor is None:
        if fallback_to_abs:
            print("Warning: Failed to load trained model, falling back to abs_res")
            return abs_res(gt, pred)
        else:
            raise RuntimeError("Failed to load trained scoring model")
    
    device = next(model.parameters()).device
    
    try:
        # Ensure tensors are properly shaped
        if pred.dim() == 1:
            # Single coordinate case - expand to [1, 1] then [1, 4]
            pred_coords = pred.unsqueeze(0).unsqueeze(0).expand(-1, 4)
            pred_scores = pred_score.unsqueeze(0) if pred_score.dim() == 0 else pred_score[:1]
            single_coord = True
        elif pred.dim() == 2 and pred.shape[1] == 4:
            # Multi-coordinate case [N, 4]
            pred_coords = pred
            pred_scores = pred_score
            single_coord = False
        else:
            raise ValueError(f"Unexpected pred shape: {pred.shape}")
        
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
            # Return scores for all samples
            return learned_scores.cpu()
    
    except Exception as e:
        if fallback_to_abs:
            print(f"Error in learned scoring: {e}, falling back to abs_res")
            return abs_res(gt, pred)
        else:
            raise RuntimeError(f"Error in learned scoring: {e}")


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
        print("Warning: Failed to load trained model, falling back to abs_res")
        return torch.abs(gt_batch - pred_batch).mean(dim=1)  # Average across coordinates
    
    device = next(model.parameters()).device
    
    # Move to device
    pred_batch = pred_batch.to(device)
    pred_score_batch = pred_score_batch.to(device)
    
    # Extract and normalize features
    features = feature_extractor.extract_features(pred_batch, pred_score_batch)
    normalized_features = feature_extractor.normalize_features(features)
    
    # Get learned scores
    with torch.no_grad():
        learned_scores = model(normalized_features).squeeze()  # [N]
    
    return learned_scores.cpu()


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
    try:
        model, feature_extractor = load_trained_scoring_model(model_path)
        return model is not None and feature_extractor is not None
    except:
        return False
