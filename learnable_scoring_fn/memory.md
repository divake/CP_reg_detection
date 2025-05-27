# Learnable Scoring Function Development Memory

## 🎯 **Project Context**

This module implements a **regression-based learnable scoring function** for conformal prediction in object detection. The goal is to replace fixed scoring functions (like `abs_res`) with adaptive ones that learn from data.

## 📋 **Development History & Key Decisions**

### **Phase 1: Problem Identification**
- **Issue**: Original approach treated scoring as classification (score <= tau)
- **Root Cause**: Fundamental misunderstanding of conformal prediction mechanics
- **Solution**: Switched to regression approach (|residual| <= width * tau)

### **Phase 2: Architecture Design**
- **Model**: `RegressionScoringFunction` outputs positive interval widths
- **Activation**: Softplus + clamping to ensure positive, bounded widths
- **Features**: 17-dimensional feature vector (geometric + uncertainty)
- **Loss**: Coverage loss + efficiency loss + calibration loss

### **Phase 3: Implementation & Validation**
- **Synthetic Data**: Successfully achieved 90% coverage with adaptive widths
- **Real Data**: Cached val predictions available, train predictions need collection
- **Integration**: Designed to work with existing conformal prediction pipeline

## 🔧 **Technical Specifications**

### **Core Components**
1. **Model**: `learnable_scoring_fn/model.py`
   - `RegressionScoringFunction` class
   - Input: 17 features, Output: 1 interval width
   - Architecture: [17] -> [256, 128, 64] -> [1]

2. **Features**: `learnable_scoring_fn/feature_utils.py`
   - `FeatureExtractor` class
   - 13 geometric + 4 uncertainty features
   - Normalized and robust to different image sizes

3. **Training**: `learnable_scoring_fn/train.py`
   - Main training pipeline for real data
   - Handles COCO train/val split properly
   - Includes calibration and evaluation

### **Data Pipeline**
```
COCO Train Set → Model Training
COCO Val Set → Split into:
  ├── Calibration (50%) → Compute tau
  └── Test (50%) → Evaluate coverage
```

### **Coverage Calculation**
```python
# Regression approach
residuals = |ground_truth - predictions|
widths = model(features)
tau = quantile(residuals / widths, 0.9)  # From calibration set
coverage = P(residuals <= widths * tau)  # On test set
```

## 🚨 **Critical Invariants (DO NOT CHANGE)**

### **1. Data Paths & Environment**
- **Python**: `/home/divake/miniconda3/envs/env_cu121/bin/python`
- **Working Dir**: `/ssd_4TB/divake/conformal-od`
- **COCO Data**: `data/coco/` (NOT `datasets/coco/`)
- **Annotations**: `data/coco/annotations/instances_train2017.json`
- **Images**: `data/coco/train2017/`

### **2. Model Configuration**
- **Checkpoint**: `checkpoints/x101fpn_train_qr_5k_postprocess.pth`
- **Config Path**: Use local detectron2 configs, not model zoo
- **ROI Head**: Use `StandardROIHeads`, not `QuantileROIHead`

### **3. Regression Framework**
- **NEVER** return to classification approach
- **ALWAYS** predict positive interval widths
- **ALWAYS** compute tau from normalized residuals: `|residual| / width`
- **ALWAYS** evaluate coverage as: `P(|residual| <= width * tau)`

### **4. Feature Engineering**
- **17 features total**: 13 geometric + 4 uncertainty
- **Normalization**: Features normalized by image dimensions
- **Uncertainty**: Include confidence scores and derived uncertainty measures

## 📊 **Validated Results**

### **Synthetic Data (train_minimal.py)**
- ✅ **Target Coverage**: 90%
- ✅ **Achieved Coverage**: 89.9% - 90.3%
- ✅ **Efficiency**: Decreasing interval widths over training
- ✅ **Tau Adaptation**: Stable tau values as model learns

### **Real Data Status**
- ✅ **Val Predictions**: Cached in `experiments/cache/real_predictions.pkl`
- ⏳ **Train Predictions**: Need to be collected (time-intensive)
- ✅ **Data Pipeline**: Code ready for train data collection

## 🔄 **Current State & Next Steps**

### **What Works**
1. **Regression Model**: `RegressionScoringFunction` validated
2. **Feature Extraction**: 17-feature pipeline working
3. **Training Logic**: Coverage + efficiency optimization
4. **Synthetic Validation**: 90% coverage achieved

### **What's Needed**
1. **Train Data Collection**: Run model on COCO train set
2. **Real Data Training**: Train on actual predictions
3. **Hyperparameter Tuning**: Optimize for real data
4. **Integration Testing**: Connect with conformal pipeline

### **Immediate Commands**
```bash
# Test current approach
cd /ssd_4TB/divake/conformal-od
/home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/train_minimal.py

# For real data (when ready)
/home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/train.py \
  --config_file cfg_std_rank --config_path config/coco_val/
```

## 🚫 **Known Issues & Solutions**

### **1. Model Zoo Errors**
- **Problem**: `RuntimeError: COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml not available`
- **Solution**: Use local config path + local checkpoint
- **Fixed In**: `train.py` lines 137-140

### **2. Data Path Errors**
- **Problem**: `FileNotFoundError: datasets/coco/annotations/...`
- **Solution**: Use `data/coco/` not `datasets/coco/`
- **Fixed In**: `train.py` lines 132-133

### **3. ROI Head Mismatch**
- **Problem**: `KeyError: "No object named 'QuantileROIHead'"`
- **Solution**: Override with `StandardROIHeads`
- **Fixed In**: `train.py` line 140

## 📁 **File Dependencies**

### **Core Files (Required)**
- `model.py`: RegressionScoringFunction class
- `feature_utils.py`: FeatureExtractor class
- `train_minimal.py`: Working synthetic data training
- `train.py`: Full real data training pipeline

### **Data Files (Required)**
- `experiments/cache/real_predictions.pkl`: Cached val predictions
- `checkpoints/x101fpn_train_qr_5k_postprocess.pth`: Model weights

### **Config Files (Required)**
- `config/coco_val/cfg_std_rank.yaml`: Val dataset config
- `config/coco_train/cfg_qr_train.yaml`: Train dataset config

## 🔍 **Debugging Guide**

### **Quick Health Check**
```bash
cd /ssd_4TB/divake/conformal-od

# 1. Check if conda env works
/home/divake/miniconda3/envs/env_cu121/bin/python --version

# 2. Check if model can be imported
/home/divake/miniconda3/envs/env_cu121/bin/python -c "from learnable_scoring_fn.model import RegressionScoringFunction; print('✓ Model imports')"

# 3. Check if training works
/home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/train_minimal.py
```

### **Common Error Patterns**
1. **Import Errors**: Check conda environment activation
2. **Path Errors**: Verify working directory is `/ssd_4TB/divake/conformal-od`
3. **CUDA Errors**: Ensure GPU is available
4. **Data Errors**: Check file paths match exactly

## 💡 **Key Insights**

1. **Regression > Classification**: Fundamental insight that changed everything
2. **Feature Engineering**: Uncertainty features are crucial for adaptive intervals
3. **Tau Calculation**: Must be from normalized residuals, not raw scores
4. **Data Splits**: Use project's standard train/val split methodology
5. **Validation**: Synthetic data testing proved the approach before real data

## 📝 **Development Notes**

- **Started**: Classification approach (wrong)
- **Pivoted**: Regression approach (correct)
- **Validated**: Synthetic data (90% coverage)
- **Status**: Ready for real data training
- **Next**: Collect train predictions and train on real data