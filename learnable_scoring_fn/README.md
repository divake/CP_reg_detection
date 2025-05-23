# Learnable Scoring Function for Conformal Object Detection

This module implements a **learnable scoring function** for conformal prediction in object detection, following a clean separation between training and usage.

## 🎯 **Overview**

Instead of using fixed scoring functions like `abs_res()`, this approach trains a neural network to learn optimal nonconformity scores that:
- **Achieve target coverage** (90%) 
- **Minimize prediction interval width**
- **Adapt to different object types and features**

## 📁 **Directory Structure**

```
learnable_scoring_fn/
├── __init__.py              # Package initialization
├── model.py                 # Neural network architecture & loss functions
├── feature_utils.py         # Feature extraction & normalization  
├── data_utils.py           # Data loading & preprocessing
├── train_scoring.py        # Main training script
├── run_training.py         # Simple training wrapper
├── trained_models/         # Saved model checkpoints
├── experiments/            # Training experiment outputs
└── README.md               # This file
```

## 🚀 **Quick Start**

### 1. **Train the Scoring Function**

```bash
cd /ssd_4TB/divake/conformal-od

# Simple training with defaults
python learnable_scoring_fn/run_training.py

# Or with custom parameters
python -m learnable_scoring_fn.train_scoring \
  --config_file cfg_std_rank \
  --subset_size 50000 \
  --num_epochs 100 \
  --target_coverage 0.9 \
  --device cuda
```

### 2. **Use Trained Model in Conformal Prediction**

```python
# In your conformal prediction code
from calibration.conformal_scores import learned_score

# Use like abs_res(), but with learned scoring
score = learned_score(gt_coords, pred_coords, pred_scores)
```

## 🔧 **Training Framework**

The training follows your **classification framework pattern**:

### **Per-Epoch Training Loop**

```python
for epoch in range(num_epochs):
    # 1. Calibration Phase: Calculate tau using calibration set
    tau = calculate_tau_per_class(model, cal_data, alpha=0.1)
    
    # 2. Training Phase: Train scoring function with FIXED tau
    train_loss = train_epoch(model, train_data, tau)
    
    # 3. Validation Phase: Evaluate with learned scores + current tau
    coverage, width = validate_epoch(model, val_data, tau)
```

### **Key Features**

- ✅ **Per-epoch tau calculation** like your classification approach
- ✅ **Curriculum learning**: λ_width starts low (focus coverage) → increases (balance coverage/efficiency)  
- ✅ **Stratified sampling**: 50k samples from top 6 COCO classes
- ✅ **Hand-crafted features**: coordinates + confidence + 6 geometric features
- ✅ **Same val/calib splits** as std method for fair comparison

## 📊 **Input Features**

The scoring function uses **13 input features**:

| Feature Type | Features | Description |
|-------------|----------|-------------|
| **Coordinates** | x0, y0, x1, y1 | Predicted bounding box coordinates |
| **Confidence** | pred_score | Model confidence score |
| **Geometric** | log_area | Log-transformed box area |
| | aspect_ratio | Width/height ratio |
| | center_x, center_y | Normalized center coordinates |
| | rel_pos_x, rel_pos_y | Position relative to image center |
| | rel_size | Box size relative to image |
| | edge_distance | Distance to nearest image edge |

## 🏗️ **Model Architecture**

```python
ScoringMLP(
    input_dim=13,           # 13 input features
    hidden_dims=[128, 64, 32],  # 3 hidden layers
    output_dim=1,           # Single score output
    dropout_rate=0.2        # Regularization
)
```

**Loss Function:**
```python
total_loss = coverage_loss + λ_width * width_loss
```

Where:
- `coverage_loss = (actual_coverage - target_coverage)²`
- `width_loss = normalized_interval_width`
- `λ_width` follows curriculum: 0.01 → 0.1

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
experiments/learnable_scoring_default/
├── best_model.pt           # Best model checkpoint
├── feature_stats.pt        # Feature normalization stats
├── training_curves.png     # Loss/coverage/width plots
├── training_metrics.json   # All training metrics
├── config.json            # Training configuration
└── checkpoints/           # Periodic checkpoints
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

After training, you should see:
- **Coverage**: Converging to ~90%
- **Width**: Decreasing over time (more efficient intervals)
- **Loss**: Generally decreasing with some fluctuation

The trained model will then be automatically used when you call `learned_score()` in your conformal prediction pipeline!

## 🎯 **Next Steps**

1. **Train the model**: Run `python learnable_scoring_fn/run_training.py`
2. **Compare with std method**: Use same evaluation scripts
3. **Analyze results**: Check if learned scoring improves coverage/efficiency trade-off
4. **Iterate**: Adjust hyperparameters based on results 