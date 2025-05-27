# Learnable Scoring Function for Conformal Object Detection

This module implements a **regression-based learnable scoring function** that predicts interval widths for conformal prediction in object detection.

## 🎯 **Overview**

### **Problem Statement**
Traditional conformal prediction uses fixed scoring functions like `abs_res()` that don't adapt to prediction uncertainty. This can lead to either:
- **Under-coverage**: Intervals too narrow for uncertain predictions
- **Inefficiency**: Intervals too wide for confident predictions

### **Solution: Regression-Based Scoring**
We train a neural network to predict **adaptive interval widths** that:
- **Achieve target coverage** (90%) by ensuring |gt - pred| <= width * tau
- **Minimize prediction interval width** for efficiency  
- **Adapt to prediction uncertainty** using both geometric and uncertainty features

### **Key Innovation**
- **Before**: Classification approach with score <= tau coverage
- **After**: Regression approach with |residual| <= width * tau coverage
- **Result**: Better coverage-efficiency trade-off through learned uncertainty estimation

## 📁 **Directory Structure**

```
learnable_scoring_fn/
├── __init__.py              # Package initialization
├── model.py                 # RegressionScoringFunction class
├── feature_utils.py         # FeatureExtractor class (17 features)
├── data_utils.py           # Data loading & preprocessing utilities
├── train.py                # Main training script (full pipeline)
├── train_minimal.py        # Minimal training with synthetic data
├── run_training.py         # Simple training wrapper
├── experiments/            # Training results and cached data
│   ├── cache/              # Cached predictions (real_predictions.pkl)
│   ├── minimal_test/       # Synthetic data training results
│   └── real_data_v1/       # Real data training results
├── README.md               # This file
├── memory.md               # Development memory and decisions
└── TRAINING_SUMMARY.md     # Summary of training approach
```

## 🚀 **Quick Start**

### **Status: ✅ Proof of Concept Complete**
The regression approach is validated with synthetic data achieving 90% coverage.

### 1. **Test with Synthetic Data (Working)**

```bash
cd /ssd_4TB/divake/conformal-od

# Run minimal training to verify the approach
/home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/train_minimal.py
```

### 2. **Train with Real Data (Next Step)**

```bash
# First collect train predictions (time-intensive)
/home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/train.py \
  --config_file cfg_std_rank \
  --config_path config/coco_val/

# Or use simplified approach with val data split
/home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/train_simplified.py
```

### 3. **Use Trained Model for Prediction Intervals**

```python
from learnable_scoring_fn.model import RegressionScoringFunction
import torch

# Load trained model
checkpoint = torch.load('experiments/minimal_test/best_model.pth')
model = RegressionScoringFunction(input_dim=10, hidden_dims=[64, 32])
model.load_state_dict(checkpoint['model_state_dict'])

# Get interval widths
widths = model(features)

# Calculate prediction intervals
tau = checkpoint['tau']
lower_bounds = predictions - widths * tau
upper_bounds = predictions + widths * tau
```

## 🔧 **Training Framework**

The training follows the **regression framework**:

### **Per-Epoch Training Loop**

```python
for epoch in range(num_epochs):
    # 1. Calibration Phase: Calculate tau from normalized residuals
    widths = model(cal_features)
    tau = quantile(|gt - pred| / widths, 0.9)
    
    # 2. Training Phase: Train to predict interval widths
    losses = criterion(widths, gt_coords, pred_coords, tau)
    
    # 3. Validation Phase: Evaluate coverage and efficiency
    coverage = P(|gt - pred| <= width * tau)
```

### **Key Features**

- ✅ **Regression-based**: Outputs interval widths, not classification scores
- ✅ **Proper tau calculation**: From normalized residuals, not raw scores  
- ✅ **Real predictions**: Uses actual model outputs from COCO validation
- ✅ **Uncertainty features**: Incorporates prediction uncertainty indicators
- ✅ **Calibration loss**: Ensures widths are proportional to errors

## 📊 **Input Features**

The scoring function uses **17 input features**:

### Geometric Features (13)
- **Coordinates**: x0, y0, x1, y1 (predicted box)
- **Confidence**: Model confidence score
- **Box properties**: log_area, aspect_ratio
- **Position**: center_x, center_y (normalized)
- **Relative**: rel_pos_x, rel_pos_y (from image center)
- **Size**: rel_size (relative to image)
- **Edge**: edge_distance (to nearest border)

### Uncertainty Features (4)
- **Confidence uncertainty**: 1 - confidence
- **Ensemble uncertainty**: Standard deviation (if available)
- **Expected error**: Based on confidence mapping
- **Difficulty score**: Based on box size and aspect ratio

## 🏗️ **Model Architecture**

```python
RegressionScoringFunction(
    input_dim=17,              # 17 input features
    hidden_dims=[256, 128, 64], # 3 hidden layers
    dropout_rate=0.15,         # Regularization
    activation='relu'          # With batch normalization
)
```

**Loss Components:**
```python
total_loss = coverage_loss + λ_efficiency * efficiency_loss + λ_calibration * calibration_loss
```

Where:
- `coverage_loss = (P(|gt-pred| <= width*tau) - 0.9)²`
- `efficiency_loss = mean(widths) / mean(errors)`
- `calibration_loss = std(errors / widths)`

## 📈 **Training Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subset_size` | 50,000 | Training samples from COCO |
| `num_epochs` | 100 | Training epochs |
| `batch_size` | 64 | Batch size |
| `target_coverage` | 0.9 | Target coverage (90%) |
| `initial_lambda` | 0.01 | Starting width penalty |
| `final_lambda` | 0.1 | Final width penalty |
| `warmup_epochs` | 20 | Epochs to keep initial λ |
| `ramp_epochs` | 30 | Epochs to ramp λ |

## 📂 **Output Files**

After training, you'll find:

```
experiments/real_data_v1/
├── best_model.pt           # Best model checkpoint
├── data_stats.pt          # Feature & error statistics
├── training_results.png    # Comprehensive plots
├── results.json           # All metrics and config
└── training.log           # Detailed training log
```

## 🔗 **Integration**

The trained model integrates seamlessly with your existing pipeline:

### **In conformal_scores.py**
```python
# Loads trained model automatically
from calibration.conformal_scores import learned_score

# Use exactly like abs_res()
learned_scores = learned_score(gt_coords, pred_coords, pred_scores)
```

### **Fallback Mechanism**
- If trained model fails to load → automatically falls back to `abs_res()`
- Ensures your pipeline never breaks

## 🎛️ **Advanced Usage**

### **Custom Training**

```bash
python -m learnable_scoring_fn.train_scoring \
  --subset_size 30000 \
  --num_epochs 50 \
  --initial_lambda 0.005 \
  --final_lambda 0.2 \
  --hidden_dims 256 128 64 \
  --dropout_rate 0.3
```

### **Load Custom Model**

```python
from calibration.conformal_scores import learned_score

# Use specific model path
score = learned_score(
    gt_coords, pred_coords, pred_scores,
    model_path="/path/to/your/model.pt"
)
```

### **Batch Processing**

```python
from calibration.conformal_scores import get_learned_score_batch

# Efficient batch processing
scores = get_learned_score_batch(gt_batch, pred_batch, score_batch)
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   python -m learnable_scoring_fn.train_scoring --batch_size 32
   ```

2. **Model not found**
   ```python
   # Check if model exists
   from calibration.conformal_scores import is_learned_model_available
   print(is_learned_model_available())  # Should return True
   ```

3. **Feature extraction errors**
   - Ensure `pred_scores` are provided when calling `learned_score()`
   - Check tensor shapes match expected formats

### **Logging**

Training logs are saved to the experiment directory. Check:
- `log.txt` for detailed training logs
- `training_metrics.json` for numerical results
- `training_curves.png` for visual progress

## 🚦 **What to Expect**

After training with real data, you should see:
- **Coverage**: Approaching 90% target
- **Average Width**: Decreasing while maintaining coverage
- **Tau**: Stabilizing as model learns appropriate widths
- **Calibration STD**: Low values indicate well-calibrated intervals

The model learns to predict wider intervals for uncertain predictions and narrower intervals for confident ones.

## 🎯 **Next Steps**

1. **Train the model**: Run `python learnable_scoring_fn/run_training.py`
2. **Compare with std method**: Use same evaluation scripts
3. **Analyze results**: Check if learned scoring improves coverage/efficiency trade-off
4. **Iterate**: Adjust hyperparameters based on results 