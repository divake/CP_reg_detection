"""
Learnable Scoring Function Package

This package implements a learnable scoring function for conformal prediction
in object detection, following the classification framework pattern with 
per-epoch tau calculation and curriculum learning.

Main Components:
- model.py: Neural network architecture and loss functions
- feature_utils.py: Feature extraction and normalization
- data_utils.py: Data loading and preprocessing utilities  
- train_scoring.py: Main training script

Usage:
1. Train the scoring function:
   ```bash
   cd /ssd_4TB/divake/conformal-od
   python -m learnable_scoring_fn.train_scoring --config_file cfg_std_rank
   ```

2. Use trained model in conformal prediction:
   ```python
   from calibration.conformal_scores import learned_score
   score = learned_score(gt_coords, pred_coords, pred_scores)
   ```
"""

__version__ = "1.0.0"
__author__ = "Conformal Object Detection Team"

# Import main components
from .model import ScoringMLP, CoverageLoss, AdaptiveLambdaScheduler, save_model, load_model
from .feature_utils import FeatureExtractor, get_feature_names
from .data_utils import (
    prepare_training_data, 
    split_data, 
    COCOClassMapper,
    get_coco_class_frequencies,
    get_top_classes
)

# Define what gets imported with "from learnable_scoring_fn import *"
__all__ = [
    # Model components
    'ScoringMLP',
    'CoverageLoss', 
    'AdaptiveLambdaScheduler',
    'save_model',
    'load_model',
    
    # Feature utilities
    'FeatureExtractor',
    'get_feature_names',
    
    # Data utilities
    'prepare_training_data',
    'split_data',
    'COCOClassMapper',
    'get_coco_class_frequencies',
    'get_top_classes',
] 