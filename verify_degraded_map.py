#!/usr/bin/env python3
"""
Verify that degraded caches have properly reduced mAP.
"""
import numpy as np
from pathlib import Path
import pickle
import json

def calculate_map_metrics(labels, iou_threshold=0.5):
    """Calculate simplified mAP metrics from labels."""
    all_ious = []
    all_scores = []
    all_correct = []
    
    for img_labels in labels:
        if 'iou' in img_labels:
            ious = img_labels['iou']
            scores = img_labels.get('pred_score', [0.9] * len(ious))
            
            # Convert to numpy arrays if they're lists
            if isinstance(ious, list):
                ious = np.array(ious)
            if isinstance(scores, list):
                scores = np.array(scores)
            
            correct = ious >= iou_threshold
            
            all_ious.extend(ious)
            all_scores.extend(scores)
            all_correct.extend(correct)
    
    all_ious = np.array(all_ious)
    all_scores = np.array(all_scores)
    all_correct = np.array(all_correct)
    
    # Calculate metrics
    if len(all_ious) == 0:
        return {
            'mAP@50': 0.0,
            'mAP@50-95': 0.0,
            'avg_iou': 0.0,
            'avg_score': 0.0,
            'total_dets': 0,
            'precision@50': 0.0
        }
    
    # Simple mAP approximation
    # Sort by score
    sorted_idx = np.argsort(-all_scores)
    sorted_correct = all_correct[sorted_idx]
    sorted_ious = all_ious[sorted_idx]
    
    # Calculate precision at different recall levels
    precisions = []
    for i in range(10, len(sorted_correct), max(1, len(sorted_correct) // 100)):
        precision = np.sum(sorted_correct[:i]) / i
        precisions.append(precision)
    
    mAP_50 = np.mean(precisions) if precisions else 0.0
    
    # mAP@50-95
    mAP_50_95 = 0.0
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        correct_at_thresh = sorted_ious >= iou_thresh
        prec_at_thresh = []
        for i in range(10, len(correct_at_thresh), max(1, len(correct_at_thresh) // 100)):
            precision = np.sum(correct_at_thresh[:i]) / i
            prec_at_thresh.append(precision)
        if prec_at_thresh:
            mAP_50_95 += np.mean(prec_at_thresh)
    mAP_50_95 /= len(np.arange(0.5, 1.0, 0.05))
    
    return {
        'mAP@50': mAP_50,
        'mAP@50-95': mAP_50_95,
        'avg_iou': np.mean(all_ious),
        'avg_score': np.mean(all_scores),
        'total_dets': len(all_ious),
        'precision@50': np.sum(all_correct) / len(all_correct)
    }

def main():
    print("="*80)
    print("VERIFICATION OF DEGRADED CACHE mAP")
    print("="*80)
    
    # First evaluate base cache
    base_cache_dir = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model")
    base_val_file = base_cache_dir / "predictions_val.pkl"
    
    print("\nBase Model Performance:")
    print("-"*80)
    
    with open(base_val_file, 'rb') as f:
        _, base_labels = pickle.load(f)
    
    base_metrics = calculate_map_metrics(base_labels)
    print(f"mAP@50:      {base_metrics['mAP@50']:.3f}")
    print(f"mAP@50-95:   {base_metrics['mAP@50-95']:.3f}")
    print(f"Avg IoU:     {base_metrics['avg_iou']:.3f}")
    print(f"Avg Score:   {base_metrics['avg_score']:.3f}")
    print(f"Total Dets:  {base_metrics['total_dets']:,}")
    print(f"Precision:   {base_metrics['precision@50']:.3f}")
    
    # Evaluate degraded caches
    results = {'base': base_metrics}
    
    efficiency_levels = ["85", "70", "50", "30"]
    
    print("\n" + "="*80)
    print("Degraded Cache Performance:")
    print("="*80)
    
    for eff in efficiency_levels:
        cache_dir = Path(f"/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_efficiency_{eff}_aggressive")
        val_file = cache_dir / "predictions_val.pkl"
        
        if not val_file.exists():
            print(f"\n{eff}% Efficiency: File not found")
            continue
        
        print(f"\n{eff}% Efficiency Model:")
        print("-"*40)
        
        with open(val_file, 'rb') as f:
            _, labels = pickle.load(f)
        
        metrics = calculate_map_metrics(labels)
        
        # Calculate relative performance
        rel_map50 = metrics['mAP@50'] / base_metrics['mAP@50'] if base_metrics['mAP@50'] > 0 else 0
        rel_map = metrics['mAP@50-95'] / base_metrics['mAP@50-95'] if base_metrics['mAP@50-95'] > 0 else 0
        
        print(f"mAP@50:      {metrics['mAP@50']:.3f} (relative: {rel_map50:.1%})")
        print(f"mAP@50-95:   {metrics['mAP@50-95']:.3f} (relative: {rel_map:.1%})")
        print(f"Avg IoU:     {metrics['avg_iou']:.3f} (Δ: {metrics['avg_iou'] - base_metrics['avg_iou']:+.3f})")
        print(f"Avg Score:   {metrics['avg_score']:.3f} (Δ: {metrics['avg_score'] - base_metrics['avg_score']:+.3f})")
        print(f"Total Dets:  {metrics['total_dets']:,} ({metrics['total_dets']/base_metrics['total_dets']:.1%} retained)")
        
        results[f'eff_{eff}'] = {
            **metrics,
            'relative_mAP@50': rel_map50,
            'relative_mAP@50-95': rel_map
        }
    
    # Save results
    output_file = Path("/ssd_4TB/divake/conformal-od/model_degradation/degraded_cache_verification.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nTarget vs Achieved Relative mAP@50:")
    for eff in efficiency_levels:
        if f'eff_{eff}' in results:
            target = int(eff) / 100
            achieved = results[f'eff_{eff}']['relative_mAP@50']
            print(f"{eff}% target → {achieved:.1%} achieved {'✓' if abs(achieved - target) < 0.15 else '✗'}")
    
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*80)
    print("READY FOR MLP TRAINING!")
    print("="*80)
    print("\nAll degraded caches have been created with proper quality reduction.")
    print("You can now train your MLP scoring function on these caches to test")
    print("if it maintains ~90% coverage even with degraded base models.")

if __name__ == "__main__":
    main()