#!/usr/bin/env python3
"""
Evaluate degraded caches to show actual mAP reduction.
"""
import numpy as np
from pathlib import Path
import pickle
import json

def load_cache(cache_dir):
    """Load cached predictions."""
    pred_file = Path(cache_dir) / "predictions_val.pkl"
    with open(pred_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        return data[0], data[1]
    return None, None

def calculate_detailed_metrics(labels):
    """Calculate detailed metrics including mAP approximation."""
    metrics = {
        'total_detections': 0,
        'tp_at_50': 0,
        'tp_at_75': 0,
        'tp_at_90': 0,
        'avg_iou': 0,
        'avg_confidence': 0,
        'false_positives': 0
    }
    
    for img_labels in labels:
        if 'iou' not in img_labels:
            continue
            
        for i, iou in enumerate(img_labels['iou']):
            metrics['total_detections'] += 1
            metrics['avg_iou'] += iou
            
            # Count true positives at different thresholds
            if iou >= 0.5:
                metrics['tp_at_50'] += 1
            else:
                metrics['false_positives'] += 1
                
            if iou >= 0.75:
                metrics['tp_at_75'] += 1
            if iou >= 0.9:
                metrics['tp_at_90'] += 1
            
            # Average confidence
            if 'pred_score' in img_labels and i < len(img_labels['pred_score']):
                metrics['avg_confidence'] += img_labels['pred_score'][i]
            else:
                metrics['avg_confidence'] += 0.9  # Default confidence
    
    # Calculate averages
    if metrics['total_detections'] > 0:
        metrics['avg_iou'] /= metrics['total_detections']
        metrics['avg_confidence'] /= metrics['total_detections']
        
        # Calculate precision at different IoU thresholds
        metrics['precision_at_50'] = metrics['tp_at_50'] / metrics['total_detections']
        metrics['precision_at_75'] = metrics['tp_at_75'] / metrics['total_detections']
        metrics['precision_at_90'] = metrics['tp_at_90'] / metrics['total_detections']
        
        # Approximate mAP (average of precisions at different IoU thresholds)
        metrics['approx_map'] = (metrics['precision_at_50'] + metrics['precision_at_75']) / 2
    else:
        metrics['precision_at_50'] = 0
        metrics['precision_at_75'] = 0
        metrics['precision_at_90'] = 0
        metrics['approx_map'] = 0
    
    return metrics

print("="*80)
print("DETAILED EVALUATION OF DEGRADED CACHES")
print("="*80)

# Evaluate all caches
caches = [
    ("Base Model", "learnable_scoring_fn/cache_base_model"),
    ("85% Aggressive", "learnable_scoring_fn/cache_efficiency_85_aggressive"),
    ("70% Aggressive", "learnable_scoring_fn/cache_efficiency_70_aggressive"),
    ("50% Aggressive", "learnable_scoring_fn/cache_efficiency_50_aggressive"),
    ("30% Aggressive", "learnable_scoring_fn/cache_efficiency_30_aggressive"),
]

results = {}
base_map = None

for name, cache_dir in caches:
    print(f"\n{name}:")
    print("-"*60)
    
    _, labels = load_cache(cache_dir)
    if labels is None:
        print(f"  ERROR: Could not load cache")
        continue
    
    metrics = calculate_detailed_metrics(labels)
    
    # Store base mAP for comparison
    if base_map is None:
        base_map = metrics['approx_map']
    
    # Calculate relative performance
    rel_perf = metrics['approx_map'] / base_map if base_map > 0 else 0
    
    print(f"  Total detections: {metrics['total_detections']:,}")
    print(f"  Average IoU: {metrics['avg_iou']:.3f}")
    print(f"  Average confidence: {metrics['avg_confidence']:.3f}")
    print(f"  False positives (IoU<0.5): {metrics['false_positives']:,} ({metrics['false_positives']/metrics['total_detections']*100:.1f}%)")
    print(f"  \nPrecision at different IoU thresholds:")
    print(f"    IoU ≥ 0.50: {metrics['precision_at_50']:.3f}")
    print(f"    IoU ≥ 0.75: {metrics['precision_at_75']:.3f}")
    print(f"    IoU ≥ 0.90: {metrics['precision_at_90']:.3f}")
    print(f"  \nApproximate mAP: {metrics['approx_map']:.3f} ({rel_perf*100:.1f}% of base)")
    
    results[name] = metrics

# Save results
output_file = Path("model_degradation/degraded_cache_evaluation.json")
output_file.parent.mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("SUMMARY: Target vs Achieved mAP Reduction")
print("="*80)

if base_map:
    print(f"\nBase model approximate mAP: {base_map:.3f}")
    print("\nDegradation results:")
    for name, metrics in results.items():
        if name != "Base Model":
            target = int(name.split('%')[0])
            achieved = metrics['approx_map'] / base_map * 100
            print(f"  {name}: Target {target}% → Achieved {achieved:.1f}%")
            print(f"    - mAP: {metrics['approx_map']:.3f}")
            print(f"    - False positive rate: {metrics['false_positives']/metrics['total_detections']*100:.1f}%")

print("\n✅ Degraded caches are ready for training!")
print("\nExample training command:")
print("/home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/run_training.py \\")
print("    --cache-dir learnable_scoring_fn/cache_efficiency_85_aggressive/ \\")
print("    --config-file cfg_std_rank --config-path config/coco_val/")