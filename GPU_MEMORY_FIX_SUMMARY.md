# GPU Memory Fix for Learnable Scoring Function

## Problem
The learnable scoring function was running out of GPU memory when both the detection model and scoring model tried to use the same GPU.

## Solution Implemented

### 1. **Automatic GPU Memory Check** (`calibration/conformal_scores.py`)
- Added `force_cpu` parameter to `load_trained_scoring_model()`
- Checks available GPU memory before loading model
- If less than 1GB free, automatically loads on CPU
- Added proper error handling for GPU memory exceptions

### 2. **OOM Exception Handling**
- Added try-except blocks in both `learned_score()` and `get_learned_score_batch()`
- If GPU OOM occurs, clears cache and retries on CPU
- Graceful fallback mechanism

### 3. **Model Loading Fix** (`learnable_scoring_fn/model.py`)
- Updated `load_regression_model()` to handle OOM during checkpoint loading
- Automatically retries on CPU if GPU loading fails

### 4. **Memory Management**
- Added `clear_model_cache()` function to free cached models
- Clears GPU cache when switching devices

## Usage

The fix works automatically - no code changes needed in calling code. The system will:

1. First try to load on GPU if available
2. Check GPU memory and switch to CPU if needed  
3. Handle OOM errors gracefully with automatic retry on CPU

## Manual Control

If you need to force CPU usage:
```python
from calibration.conformal_scores import load_trained_scoring_model
model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(force_cpu=True)
```

To clear cached models:
```python
from calibration.conformal_scores import clear_model_cache
clear_model_cache()
```

## Remaining Consideration

If the input tensors are already on GPU and the model is on CPU (or vice versa), tensor operations will automatically handle device placement. PyTorch will move tensors as needed, though this may impact performance slightly.