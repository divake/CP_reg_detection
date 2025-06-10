# Model Degradation Scripts

This directory contains scripts for creating degraded cache versions to test scoring function robustness.

## Purpose

Test whether the learnable scoring function maintains ~90% coverage even when trained on degraded base model outputs.

## Scripts

### 1. `scripts/degrade_all_caches_v2.py`
Main script to create degraded cache versions for all data splits (train + val).

**Usage:**
```bash
python model_degradation/scripts/degrade_all_caches_v2.py
```

Creates 4 degraded cache directories:
- `cache_efficiency_85_aggressive/` - Mild degradation (96.6% mAP retention)
- `cache_efficiency_70_aggressive/` - Moderate degradation (46.3% mAP retention)
- `cache_efficiency_50_aggressive/` - Severe degradation (~0% mAP)
- `cache_efficiency_30_aggressive/` - Extreme degradation (~0% mAP)

### 2. `scripts/degrade_cache_aggressive.py`
Alternative script for aggressive degradation (validation only).

### 3. `../verify_degraded_map.py`
Verifies the actual mAP reduction in degraded caches.

**Usage:**
```bash
python verify_degraded_map.py
```

### 4. `../evaluate_degraded_caches.py`
Detailed evaluation of degraded cache metrics.

**Usage:**
```bash
python evaluate_degraded_caches.py
```

## Degradation Method

The degradation applies to labels only (feature vectors unchanged):
- **IoU reduction**: Multiply by degradation factor + add noise
- **Score reduction**: Scale confidence scores down
- **Detection dropping**: Randomly remove detections
- **Coordinate noise**: Add gaussian noise to bounding boxes
- **False positives**: Add low-quality spurious detections

## Training with Degraded Caches

After creating degraded caches, train your scoring function:

```bash
# Example: Train on 85% degraded cache
python learnable_scoring_fn/run_training.py \
    --cache-dir learnable_scoring_fn/cache_efficiency_85_aggressive/ \
    --config-file cfg_std_rank \
    --config-path config/coco_val/
```

## Results

Degradation verification results are saved to:
- `degraded_cache_verification.json` - mAP metrics for each degradation level