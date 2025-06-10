#!/usr/bin/env python3
"""
Aggressively degrade cached predictions to achieve target mAP reduction.
This degrades quality, not just quantity.
"""
import numpy as np
from pathlib import Path
import pickle
import json
from copy import deepcopy

def load_base_cache(cache_dir="/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"):
    """Load base model cached predictions."""
    cache_path = Path(cache_dir)
    pred_file = cache_path / "predictions_val.pkl"
    
    with open(pred_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, tuple) and len(data) == 2:
        predictions, labels = data
    else:
        raise ValueError(f"Unexpected format: {type(data)}")
    
    print(f"Loaded cache with {len(predictions)} images")
    return predictions, labels

def aggressive_degrade(predictions, labels, target_efficiency):
    """Aggressively degrade predictions to achieve target mAP."""
    degraded_preds = []
    degraded_labels = []
    
    np.random.seed(42 + int(target_efficiency * 100))
    
    # More aggressive parameters
    keep_rate = 0.7 + 0.3 * target_efficiency  # Still drop some detections
    iou_scale = 0.5 + 0.5 * target_efficiency  # Scale down IoU values
    box_noise_std = 20 * (1 - target_efficiency)  # Add noise to boxes
    false_positive_rate = 0.3 * (1 - target_efficiency)  # Add false positives
    
    print(f"\nDegrading to {target_efficiency*100}% target mAP:")
    print(f"  - Keep rate: {keep_rate:.2f}")
    print(f"  - IoU scale: {iou_scale:.2f}")
    print(f"  - Box noise std: {box_noise_std:.1f} pixels")
    print(f"  - False positive rate: {false_positive_rate:.2f}")
    
    for img_idx, (img_preds, img_labels) in enumerate(zip(predictions, labels)):
        if 'iou' not in img_labels or len(img_labels['iou']) == 0:
            degraded_preds.append(img_preds)
            degraded_labels.append(img_labels)
            continue
        
        n_dets = len(img_labels['iou'])
        
        # Decide which detections to keep
        keep_mask = np.random.random(n_dets) < keep_rate
        if not any(keep_mask):
            keep_mask[0] = True
        
        # Create degraded version
        deg_labels = {}
        
        for key, value in img_labels.items():
            if isinstance(value, list) and len(value) == n_dets:
                filtered = []
                
                for i, keep in enumerate(keep_mask):
                    if not keep:
                        continue
                    
                    v = value[i]
                    
                    # Degrade based on key type
                    if key == 'iou':
                        # Aggressively reduce IoU
                        # Some detections become false positives
                        if np.random.random() < false_positive_rate:
                            v = np.random.uniform(0.1, 0.4)  # False positive
                        else:
                            v = v * iou_scale  # Scale down good detections
                            v = max(0.1, v)  # Keep minimum IoU
                    
                    elif key == 'pred_score':
                        # Reduce confidence more for degraded detections
                        v = v * (0.5 + 0.5 * target_efficiency)
                    
                    elif key in ['pred_x0', 'pred_y0', 'pred_x1', 'pred_y1']:
                        # Add noise to predicted boxes
                        v = v + np.random.normal(0, box_noise_std)
                    
                    elif key in ['gt_x0', 'gt_y0', 'gt_x1', 'gt_y1']:
                        # Don't modify ground truth
                        pass
                    
                    filtered.append(v)
                
                # Add some pure false positives
                if key == 'iou' and np.random.random() < false_positive_rate:
                    n_false = np.random.randint(1, 4)
                    for _ in range(n_false):
                        filtered.append(np.random.uniform(0.05, 0.3))
                
                deg_labels[key] = filtered
            else:
                deg_labels[key] = value
        
        # Update predictions list accordingly
        deg_preds = []
        if isinstance(img_preds, list):
            n_keep = sum(keep_mask)
            deg_preds = img_preds[:n_keep]  # Simplified
        else:
            deg_preds = img_preds
        
        degraded_preds.append(deg_preds)
        degraded_labels.append(deg_labels)
    
    return degraded_preds, degraded_labels

def calculate_simple_map(labels):
    """Quick mAP calculation for verification."""
    total_dets = 0
    correct_dets = 0
    
    for img_labels in labels:
        if 'iou' in img_labels:
            for iou in img_labels['iou']:
                total_dets += 1
                if iou >= 0.5:
                    correct_dets += 1
    
    if total_dets == 0:
        return 0
    
    # Simple approximation
    precision = correct_dets / total_dets
    return precision * 0.9  # Rough mAP approximation

def save_degraded_cache(predictions, labels, output_dir):
    """Save degraded predictions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pred_file = output_path / "predictions_val.pkl"
    with open(pred_file, 'wb') as f:
        pickle.dump((predictions, labels), f)
    
    print(f"Saved to: {output_path}")

def main():
    print("="*70)
    print("AGGRESSIVE CACHE DEGRADATION FOR TARGET mAP")
    print("="*70)
    
    # Load base cache
    base_preds, base_labels = load_base_cache()
    
    # Calculate base performance
    base_map = calculate_simple_map(base_labels)
    print(f"\nBase approximate mAP: {base_map:.3f}")
    
    # Target mAP levels (as fraction of base)
    targets = [
        (0.85, "efficiency_85_aggressive"),
        (0.70, "efficiency_70_aggressive"),
        (0.50, "efficiency_50_aggressive"),
        (0.30, "efficiency_30_aggressive")
    ]
    
    for target, name in targets:
        # Degrade more aggressively
        deg_preds, deg_labels = aggressive_degrade(base_preds, base_labels, target)
        
        # Check degradation
        deg_map = calculate_simple_map(deg_labels)
        print(f"\n{name}: approx mAP = {deg_map:.3f} ({deg_map/base_map*100:.1f}% of base)")
        
        # Save
        output_dir = f"/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_{name}"
        save_degraded_cache(deg_preds, deg_labels, output_dir)
    
    print("\n" + "="*70)
    print("Created aggressive degradation caches:")
    for _, name in targets:
        print(f"  - cache_{name}/")
    print("\nThese should have properly reduced mAP!")

if __name__ == "__main__":
    main()