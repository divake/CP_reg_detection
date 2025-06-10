#!/usr/bin/env python3
"""
Degrade cached predictions - Version 2 that handles the actual cache format.
The cache contains feature vectors and label dictionaries with detection info.
"""
import numpy as np
from pathlib import Path
import pickle
import json
from copy import deepcopy

def degrade_labels(labels, target_efficiency):
    """
    Degrade label dictionaries to simulate a weaker model.
    
    Args:
        labels: List of label dictionaries
        target_efficiency: Target efficiency (e.g., 0.85 for 85%)
    
    Returns:
        degraded_labels
    """
    np.random.seed(42)  # For reproducibility
    
    degraded_labels = []
    
    # Calculate degradation parameters
    iou_degradation = 1.0 - (1.0 - target_efficiency) * 1.5  # Aggressive IoU reduction
    score_scale = target_efficiency ** 0.7  # Moderate score reduction
    drop_rate = (1.0 - target_efficiency) * 0.3  # Drop some detections
    
    for img_labels in labels:
        # Work with a copy
        new_labels = deepcopy(img_labels)
        
        # Get number of detections (from IoU list length)
        if 'iou' not in new_labels:
            degraded_labels.append(new_labels)
            continue
            
        n_dets = len(new_labels['iou'])
        if n_dets == 0:
            degraded_labels.append(new_labels)
            continue
        
        # 1. Randomly drop some detections
        keep_mask = np.random.random(n_dets) > drop_rate
        indices_to_keep = np.where(keep_mask)[0]
        
        if len(indices_to_keep) == 0:
            # Keep at least 10% of detections
            n_keep = max(1, int(n_dets * 0.1))
            indices_to_keep = np.random.choice(n_dets, n_keep, replace=False)
        
        # Filter all label fields
        for key in new_labels.keys():
            if isinstance(new_labels[key], list) and len(new_labels[key]) == n_dets:
                new_labels[key] = [new_labels[key][i] for i in indices_to_keep]
            elif isinstance(new_labels[key], np.ndarray) and len(new_labels[key]) == n_dets:
                new_labels[key] = new_labels[key][indices_to_keep]
        
        # 2. Degrade IoU values
        if 'iou' in new_labels:
            degraded_ious = []
            for iou in new_labels['iou']:
                # Apply degradation factor
                new_iou = iou * iou_degradation
                # Add noise
                noise = np.random.normal(0, 0.03 * (1 - target_efficiency))
                new_iou = max(0, min(1, new_iou + noise))
                degraded_ious.append(new_iou)
            new_labels['iou'] = degraded_ious
        
        # 3. Reduce confidence scores
        if 'pred_score' in new_labels:
            degraded_scores = []
            for score in new_labels['pred_score']:
                new_score = score * score_scale
                # Add some noise
                noise = np.random.normal(0, 0.02 * (1 - target_efficiency))
                new_score = max(0, min(1, new_score + noise))
                degraded_scores.append(new_score)
            new_labels['pred_score'] = degraded_scores
        
        # 4. Add noise to bounding box coordinates
        coord_noise_scale = 5.0 * (1 - target_efficiency)  # pixels
        for coord_key in ['pred_x0', 'pred_y0', 'pred_x1', 'pred_y1']:
            if coord_key in new_labels:
                degraded_coords = []
                for coord in new_labels[coord_key]:
                    noise = np.random.normal(0, coord_noise_scale)
                    degraded_coords.append(coord + noise)
                new_labels[coord_key] = degraded_coords
        
        # 5. Add some false positives (low quality detections)
        if np.random.random() < (1 - target_efficiency) * 0.5:
            n_false = max(1, int(len(indices_to_keep) * 0.1 * (1 - target_efficiency)))
            
            # For false positives, duplicate some existing detections but corrupt them heavily
            false_indices = np.random.choice(len(new_labels['iou']), n_false, replace=True)
            
            for key in new_labels.keys():
                if isinstance(new_labels[key], list) and len(new_labels[key]) > 0:
                    # Add corrupted versions
                    false_values = []
                    for idx in false_indices:
                        if key == 'iou':
                            # Very low IoU for false positives
                            false_values.append(np.random.uniform(0, 0.3))
                        elif key == 'pred_score':
                            # Low confidence
                            false_values.append(np.random.uniform(0.3, 0.5) * score_scale)
                        elif key in ['pred_x0', 'pred_y0', 'pred_x1', 'pred_y1']:
                            # Add large noise to coordinates
                            false_values.append(new_labels[key][idx] + np.random.normal(0, 20))
                        else:
                            # Copy existing value
                            false_values.append(new_labels[key][idx])
                    
                    new_labels[key].extend(false_values)
        
        degraded_labels.append(new_labels)
    
    return degraded_labels

def degrade_cache_file(input_file, output_file, target_efficiency):
    """Degrade a single cache file."""
    print(f"  Loading {input_file.name}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, tuple) and len(data) == 2:
        predictions, labels = data
    elif isinstance(data, dict):
        predictions = data.get('predictions', data.get('preds', None))
        labels = data.get('labels', data.get('matches', None))
    else:
        raise ValueError(f"Unexpected data format in {input_file}")
    
    print(f"  Loaded {len(predictions)} images")
    print(f"  Degrading labels to {target_efficiency*100:.0f}% efficiency...")
    
    # We keep predictions (feature vectors) unchanged
    # Only degrade the labels (detection info)
    degraded_labels = degrade_labels(labels, target_efficiency)
    
    # Calculate some statistics
    original_dets = sum(len(l.get('iou', [])) for l in labels)
    degraded_dets = sum(len(l.get('iou', [])) for l in degraded_labels)
    
    print(f"  Original detections: {original_dets}")
    print(f"  Degraded detections: {degraded_dets} ({degraded_dets/original_dets*100:.1f}%)")
    
    # Save in the same format as loaded
    print(f"  Saving to {output_file.name}...")
    with open(output_file, 'wb') as f:
        pickle.dump((predictions, degraded_labels), f)
    
    return len(predictions)

def main():
    """Main function to degrade all cache files."""
    print("="*70)
    print("COMPREHENSIVE CACHE DEGRADATION (V2) FOR ALL DATA SPLITS")
    print("="*70)
    
    # Base cache directory
    base_cache_dir = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model")
    
    # Efficiency levels to create
    efficiency_levels = [
        ("85", 0.85),
        ("70", 0.70),
        ("50", 0.50),
        ("30", 0.30)
    ]
    
    # Cache files to degrade
    cache_files = ["predictions_train.pkl", "predictions_val.pkl", "predictions_test.pkl"]
    
    for eff_name, eff_value in efficiency_levels:
        print(f"\nCreating {eff_name}% efficiency caches...")
        print("-"*70)
        
        # Create output directory (overwrite existing)
        output_dir = Path(f"/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_efficiency_{eff_name}_aggressive")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Track statistics
        total_images = 0
        
        # Process each cache file
        for cache_file in cache_files:
            input_file = base_cache_dir / cache_file
            output_file = output_dir / cache_file
            
            if not input_file.exists():
                print(f"  WARNING: {cache_file} not found, skipping...")
                continue
            
            n_images = degrade_cache_file(input_file, output_file, eff_value)
            total_images += n_images
        
        # Save degradation info
        info = {
            'target_efficiency': eff_value,
            'efficiency_percentage': eff_name,
            'base_cache_dir': str(base_cache_dir),
            'total_images_processed': total_images,
            'degradation_method': 'aggressive_v2',
            'files_degraded': [f for f in cache_files if (base_cache_dir / f).exists()],
            'degradation_details': {
                'iou_degradation': f"{(1.0 - (1.0 - eff_value) * 1.5)*100:.1f}%",
                'score_scale': f"{(eff_value ** 0.7)*100:.1f}%",
                'drop_rate': f"{(1.0 - eff_value) * 0.3*100:.1f}%",
                'coord_noise_pixels': f"{5.0 * (1 - eff_value):.1f}"
            }
        }
        
        with open(output_dir / 'degradation_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"  Created {eff_name}% efficiency cache with {total_images} total images")
    
    print("\n" + "="*70)
    print("CACHE DEGRADATION COMPLETE!")
    print("="*70)
    print("\nCreated degraded caches for ALL data splits:")
    for eff_name, _ in efficiency_levels:
        cache_dir = f"cache_efficiency_{eff_name}_aggressive"
        print(f"  - {cache_dir}/")
        print(f"      ├── predictions_train.pkl (feature vectors unchanged)")
        print(f"      ├── predictions_val.pkl  (labels degraded)")
        print(f"      ├── predictions_test.pkl  (for scoring)")
        print(f"      └── degradation_info.json")
    
    print("\nDegradation applied to labels only (IoU, scores, coordinates)")
    print("Feature vectors remain unchanged for MLP training")
    
    print("\nYou can now train your MLP on these degraded caches!")
    print("\nExample command:")
    print("  /home/divake/miniconda3/envs/env_cu121/bin/python learnable_scoring_fn/run_training.py \\")
    print("      --cache-dir learnable_scoring_fn/cache_efficiency_85_aggressive/ \\")
    print("      --config-file cfg_std_rank --config-path config/coco_val/")

if __name__ == "__main__":
    main()