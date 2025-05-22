# Learnable Scoring Function for Conformal Object Detection

This module implements a learnable scoring function for conformal prediction in object detection. It builds upon the existing conformal prediction framework but replaces handcrafted nonconformity scores with a neural network that learns to predict optimal scores for better prediction intervals.

## Overview

The learnable scoring function trains an MLP to predict nonconformity scores that result in well-calibrated prediction intervals with high coverage and minimal width. The key advantages of this approach are:

1. **Adaptive Scoring**: The model learns which features are most predictive of uncertainty for each coordinate
2. **Optimized Intervals**: Directly optimizes for both coverage and efficiency
3. **Leverages Existing Framework**: Builds on the standard conformal prediction pipeline

## Components

- `model.py`: Contains the MLP architecture and loss functions
- `learnable_conformal.py`: Main controller class implementing the risk control procedure
- `train_learnable.py`: Script for training and evaluating the model
- `update_plots.py`: Utility to update the plots.py file to support comparison with the learnable method

## Usage

### Training the Model

To train the learnable scoring function on the COCO validation dataset:

```bash
cd /ssd_4TB/divake/conformal-od
python -m learnable_scoring_fn.train_learnable \
  --config_file cfg_std_rank \
  --config_path /ssd_4TB/divake/conformal-od/config/coco_val \
  --device cuda
```

This will:
1. Collect predictions from the base model
2. Split the data into train/calibration/validation sets
3. Train the MLP to predict optimal scoring values
4. Run the conformal calibration procedure
5. Evaluate the performance of the method

### Adding to Comparison Plots

To update the `plots.py` file to include the learnable scoring function in comparison plots:

```bash
python -m learnable_scoring_fn.update_plots
```

This script will:
1. Create a backup of the original `plots.py`
2. Update the functions to support the learnable scoring function
3. Add imports for the necessary modules

After updating, you can run the multi-method comparison plots with:

```bash
cd /ssd_4TB/divake/conformal-od
python -m plots.plots
```

Or from a Python session/notebook:

```python
from plots.plots import plot_multi_method_comparison

# Compare all methods
plot_multi_method_comparison(
    img_name="000000054593", 
    class_name="person", 
    dataset="coco_val", 
    device="cuda:0", 
    to_file=True
)

# Compare specific methods
plot_multi_method_comparison(
    img_name="000000054593", 
    class_name="person", 
    dataset="coco_val", 
    device="cuda:0", 
    to_file=True,
    methods=["std", "learn"]  # Only compare standard and learnable methods
)
```

## Implementation Details

The learnable scoring function follows these steps:

1. **Data Collection**: Collects predictions from the base object detection model
2. **Feature Extraction**: Extracts features for the MLP (box coordinates, confidence scores)
3. **Training**:
   - Trains the MLP to predict nonconformity scores
   - Uses a loss function that balances coverage and interval width
   - Recalculates tau (quantile threshold) after each epoch
4. **Calibration**: Uses a calibration set to find the optimal quantile (tau) for the desired coverage level
5. **Prediction**: Applies the learned scoring function to create prediction intervals

The output format matches the existing controllers, making it compatible with the rest of the codebase.

## Results Interpretation

After training, the model will generate several plots in the output directory:

- `loss_curve.png`: Training and validation loss over epochs
- `coverage.png`: Coverage metric over epochs
- `width.png`: Prediction interval width over epochs
- `coverage_width.png`: Combined plot showing both metrics

These plots help visualize the learning progress and the trade-off between coverage and efficiency.

## File Structure

```
learnable_scoring_fn/
├── __init__.py             # Module initialization
├── model.py                # MLP architecture and loss functions
├── learnable_conformal.py  # Main controller class
├── train_learnable.py      # Training script
├── update_plots.py         # Plot update utility
└── README.md               # This file
```

## Future Improvements

Potential improvements for the learnable scoring function include:

1. More complex architectures like transformers for better feature extraction
2. End-to-end training with the base model
3. Exploration of different loss functions and regularization techniques
4. Integration with more diverse input features (e.g., image context) 