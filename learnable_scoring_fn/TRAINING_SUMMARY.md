# Learnable Scoring Function Training Summary

## Key Findings

### 1. Regression vs Classification Approach
- **Previous (Classification)**: Treated scoring as binary classification where score <= tau means covered
- **New (Regression)**: Predicts interval widths where |gt - pred| <= width * tau provides coverage
- **Result**: Regression approach successfully achieves ~90% coverage on synthetic data

### 2. Training Results
- **Synthetic Data Test**: 
  - Target coverage: 90%
  - Achieved coverage: 89.9% - 90.3%
  - Model learns to predict adaptive interval widths based on features
  - Tau values increase during training as the model learns tighter widths

### 3. Implementation Details
- **Model**: `RegressionScoringFunction` outputs positive interval widths using softplus activation
- **Loss**: Combination of coverage loss (main) and efficiency loss (regularization)
- **Calibration**: Tau computed as quantile of normalized residuals |gt - pred| / width

### 4. Next Steps for Real Data
1. **Data Collection**: Need to collect predictions from COCO train set (currently only have val cached)
2. **Feature Engineering**: Current features include:
   - Geometric features (box size, aspect ratio, position)
   - Uncertainty features (objectness score, score variance if ensemble)
   - Relative features (distance to image edges, relative size)

3. **Training Strategy**:
   - Use COCO train for training the scoring function
   - Split COCO val into calibration (compute tau) and test (evaluate coverage)
   - This matches the project's standard evaluation approach

### 5. Challenges Addressed
- Fixed data path issues (datasets vs data directory)
- Handled model loading with local checkpoints
- Resolved ROI head mismatch (QuantileROIHead vs StandardROIHeads)
- Created minimal working example to validate approach

### 6. Code Organization
- `model.py`: Regression-based scoring function
- `feature_utils.py`: Feature extraction from predictions
- `train_minimal.py`: Working example with synthetic data
- `train.py`: Full implementation for real data (needs prediction collection)

## Conclusion
The regression-based approach for learnable scoring functions is validated and working. The main remaining task is to efficiently collect predictions from the COCO training set to train the model on real data.