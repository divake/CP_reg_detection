"""
Learnable Scoring Function Package - Regression Based

This package implements a regression-based learnable scoring function for 
conformal prediction in object detection. The scoring function outputs
interval widths, not classification scores.

Main Components:
- model.py: Regression neural network and loss functions
- feature_utils.py: Feature extraction and normalization
- data_utils.py: Data loading and preprocessing utilities  
- train.py: Main training script

Usage:
1. Train the scoring function:
   ```bash
   cd /ssd_4TB/divake/conformal-od
   python learnable_scoring_fn/run_training.py
   ```

2. Use trained model for prediction intervals:
   ```python
   from learnable_scoring_fn import RegressionScoringFunction, load_regression_model
   model, checkpoint = load_regression_model('path/to/model.pt')
   widths = model(features)
   intervals = predictions ± (widths * tau)
   ```
"""

__version__ = "2.0.0"
__author__ = "Conformal Object Detection Team"

# Import main components
from .model import (
    RegressionScoringFunction,
    RegressionCoverageLoss,
    calculate_tau_regression,
    UncertaintyFeatureExtractor,
    save_regression_model,
    load_regression_model
)
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
    'RegressionScoringFunction',
    'RegressionCoverageLoss',
    'calculate_tau_regression',
    'UncertaintyFeatureExtractor',
    'save_regression_model',
    'load_regression_model',
    
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