#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various Base Model Performance Analysis Script

This script analyzes coverage and MPIW performance across 14 different base models
using standard conformal prediction, CQR, and ensemble methods.

Base Models Analyzed:
1. ResNet-50-C4 (No FPN)
2. ResNet-50-FPN  
3. ResNet-101-C4 (No FPN)
4. ResNet-101-FPN
5. ResNeXt-101-FPN
6. ResNeXt-101-FPN (BDD100K)
7. ResNeXt-101-FPN (FP16)
8. ResNeXt-101-FPN (INT8)
9. ResNeXt-101-FPN (FP16 Aggressive)
10. ResNeXt-101-FPN (CQR BDD100K)
11. ResNeXt-101-FPN (Ensemble BDD100K)
12. ResNeXt-101-FPN (Cityscapes)
13. ResNeXt-101-FPN (CQR Cityscapes)
14. ResNeXt-101-FPN (Ensemble Cityscapes)

Output: Simple coverage and MPIW statistics for each base model
"""

import os
import torch
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base model directories and configurations - COCO & BDD100K
BASE_MODEL_CONFIGS = {
    # COCO Models
    'x101fpn': {
        'model_name': 'ResNeXt-101-FPN',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_x101fpn_std_rank_class',
        'data_file': 'std_conf_x101fpn_std_rank_class_box_set.pt'
    },
    'r50fpn': {
        'model_name': 'ResNet-50-FPN',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_r50fpn_std_rank_class',
        'data_file': 'std_conf_r50fpn_std_rank_class_box_set.pt'
    },
    'r50c4': {
        'model_name': 'ResNet-50-C4 (No FPN)',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_r50c4_std_rank_class',
        'data_file': 'std_conf_r50c4_std_rank_class_box_set.pt'
    },
    'cascade_r50fpn': {
        'model_name': 'Cascade R-CNN ResNet-50-FPN',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_cascade_r50fpn_std_rank_class',
        'data_file': 'std_conf_cascade_r50fpn_std_rank_class_box_set.pt'
    },
    'r50fpn_ens': {
        'model_name': 'ResNet-50-FPN (Ensemble)',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/ens_conf_r50fpn_ens_rank_class',
        'data_file': 'ens_conf_r50fpn_ens_rank_class_box_set.pt'
    },
    'x101fpn_cqr': {
        'model_name': 'ResNeXt-101-FPN (CQR)',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/cqr_conf_x101fpn_cqr_rank_class',
        'data_file': 'cqr_conf_x101fpn_cqr_rank_class_box_set.pt'
    },
    
    # BDD100K Models
    'x101fpn_cqr_bdd100k': {
        'model_name': 'ResNeXt-101-FPN (CQR BDD100K)',
        'directory': '/ssd_4TB/divake/conformal-od/output/bdd100k_val/cqr_conf_x101fpn_cqr_rank_bdd100k',
        'data_file': 'cqr_conf_x101fpn_cqr_rank_bdd100k_box_set.pt'
    },
    'x101fpn_ens_bdd100k': {
        'model_name': 'ResNeXt-101-FPN (Ensemble BDD100K)',
        'directory': '/ssd_4TB/divake/conformal-od/output/bdd100k_val/ens_conf_x101fpn_ens_rank_bdd100k',
        'data_file': 'ens_conf_x101fpn_ens_rank_bdd100k_box_set.pt'
    },
    'cascade_r50fpn_bdd100k': {
        'model_name': 'Cascade R-CNN ResNet-50-FPN (BDD100K)',
        'directory': '/ssd_4TB/divake/conformal-od/output/bdd100k_val/std_conf_cascade_r50fpn_std_rank_bdd100k_cascade_r50fpn',
        'data_file': 'std_conf_cascade_r50fpn_std_rank_bdd100k_cascade_r50fpn_box_set.pt'
    },
    'r50c4_bdd100k': {
        'model_name': 'ResNet-50-C4 (No FPN BDD100K)',
        'directory': '/ssd_4TB/divake/conformal-od/output/bdd100k_val/std_conf_r50c4_std_rank_bdd100k_r50c4',
        'data_file': 'std_conf_r50c4_std_rank_bdd100k_r50c4_box_set.pt'
    },
    'r50fpn_bdd100k': {
        'model_name': 'ResNet-50-FPN (BDD100K)',
        'directory': '/ssd_4TB/divake/conformal-od/output/bdd100k_val/std_conf_r50fpn_std_rank_bdd100k_r50fpn',
        'data_file': 'std_conf_r50fpn_std_rank_bdd100k_r50fpn_box_set.pt'
    },
    'x101fpn_fixed_bdd100k': {
        'model_name': 'ResNeXt-101-FPN (Fixed BDD100K)',
        'directory': '/ssd_4TB/divake/conformal-od/output/bdd100k_val/std_conf_x101fpn_std_rank_fixed_bdd100k',
        'data_file': 'std_conf_x101fpn_std_rank_fixed_bdd100k_box_set.pt'
    },
    
    # Cityscapes Models
    'x101fpn_cqr_cityscapes': {
        'model_name': 'ResNeXt-101-FPN (CQR Cityscapes)',
        'directory': '/ssd_4TB/divake/conformal-od/output/cityscapes_val/cqr_conf_x101fpn_cqr_rank_cityscapes',
        'data_file': 'cqr_conf_x101fpn_cqr_rank_cityscapes_box_set.pt'
    },
    'x101fpn_ens_cityscapes': {
        'model_name': 'ResNeXt-101-FPN (Ensemble Cityscapes)',
        'directory': '/ssd_4TB/divake/conformal-od/output/cityscapes_val/ens_conf_x101fpn_ens_rank_cityscapes',
        'data_file': 'ens_conf_x101fpn_ens_rank_cityscapes_box_set.pt'
    },
    'cascade_r50fpn_cityscapes': {
        'model_name': 'Cascade R-CNN ResNet-50-FPN (Cityscapes)',
        'directory': '/ssd_4TB/divake/conformal-od/output/cityscapes_val/std_conf_cascade_r50fpn_std_rank_cityscapes_cascade_r50fpn',
        'data_file': 'std_conf_cascade_r50fpn_std_rank_cityscapes_cascade_r50fpn_box_set.pt'
    },
    'r50c4_cityscapes': {
        'model_name': 'ResNet-50-C4 (No FPN Cityscapes)',
        'directory': '/ssd_4TB/divake/conformal-od/output/cityscapes_val/std_conf_r50c4_std_rank_cityscapes_r50c4',
        'data_file': 'std_conf_r50c4_std_rank_cityscapes_r50c4_box_set.pt'
    },
    'r50fpn_cityscapes': {
        'model_name': 'ResNet-50-FPN (Cityscapes)',
        'directory': '/ssd_4TB/divake/conformal-od/output/cityscapes_val/std_conf_r50fpn_std_rank_cityscapes_r50fpn',
        'data_file': 'std_conf_r50fpn_std_rank_cityscapes_r50fpn_box_set.pt'
    },
    'x101fpn_class_cityscapes': {
        'model_name': 'ResNeXt-101-FPN (Class Cityscapes)',
        'directory': '/ssd_4TB/divake/conformal-od/output/cityscapes_val/std_conf_x101fpn_std_rank_class_cityscapes',
        'data_file': 'std_conf_x101fpn_std_rank_class_cityscapes_box_set.pt'
    }
}

# Metric indices in the tensor data
METRIC_INDICES = {
    'cov_box': 5,    # Overall box coverage
    'mpiw': 2,       # Mean Prediction Interval Width
}

# ============================================================================
# DATA LOADING AND ANALYSIS
# ============================================================================

def load_model_performance(model_id, config):
    """
    Load performance data for a single base model.
    
    Args:
        model_id (str): Model identifier (e.g., 'r50c4')
        config (dict): Model configuration
        
    Returns:
        dict: Performance metrics or None if file not found
    """
    file_path = os.path.join(config['directory'], config['data_file'])
    
    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return None
    
    try:
        # Load tensor data
        # Shape: [n_trials, n_classes, n_score_indices, n_metrics]
        control_data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Determine dataset type and valid classes
        if 'cityscapes' in model_id.lower():
            # Cityscapes only has 7 classes: person(0), bicycle(1), car(2), motorcycle(3), bus(5), train(6), truck(7)
            valid_classes = [0, 1, 2, 3, 5, 6, 7]
            dataset_type = 'cityscapes'
        elif 'bdd100k' in model_id.lower():
            # BDD100K has 9 classes: person(0), rider(0), car(2), truck(7), bus(5), train(6), motorcycle(3), bicycle(1), traffic light(9)
            valid_classes = [0, 1, 2, 3, 5, 6, 7, 9]  # Missing class 4 and 8
            dataset_type = 'bdd100k'
        else:
            # COCO has all 80 classes
            valid_classes = list(range(control_data.shape[1]))
            dataset_type = 'coco'
        
        # Extract metrics - average only over valid classes
        coverage_by_class = control_data[:, :, :, METRIC_INDICES['cov_box']].mean(dim=2)  # Average over score indices
        mpiw_by_class = control_data[:, :, 0, METRIC_INDICES['mpiw']]  # Use first score index only
        
        # For each trial, compute mean only over valid classes
        coverage_trials = []
        mpiw_trials = []
        
        for trial_idx in range(control_data.shape[0]):
            # Get coverage for valid classes only
            valid_coverages = [coverage_by_class[trial_idx, cls].item() for cls in valid_classes if cls < control_data.shape[1]]
            valid_mpiws = [mpiw_by_class[trial_idx, cls].item() for cls in valid_classes if cls < control_data.shape[1]]
            
            # Compute mean only over valid (non-zero) values
            valid_coverages = [c for c in valid_coverages if c > 0]  # Filter out zero coverages
            valid_mpiws = [m for m in valid_mpiws if m > 0]  # Filter out zero MPIWs
            
            if valid_coverages:
                coverage_trials.append(np.mean(valid_coverages))
            if valid_mpiws:
                mpiw_trials.append(np.mean(valid_mpiws))
        
        coverage_data = np.array(coverage_trials)
        mpiw_data = np.array(mpiw_trials)
        
        # Check if we have valid data
        if len(coverage_data) == 0 or len(mpiw_data) == 0:
            print(f"  ⚠️  No valid data found (all zeros or empty)")
            return None
            
        return {
            'coverage_mean': coverage_data.mean(),
            'coverage_std': coverage_data.std(),
            'coverage_min': coverage_data.min(),
            'coverage_max': coverage_data.max(),
            'mpiw_mean': mpiw_data.mean(),
            'mpiw_std': mpiw_data.std(),
            'mpiw_min': mpiw_data.min(),
            'mpiw_max': mpiw_data.max(),
            'n_trials': control_data.shape[0],
            'n_classes': len(valid_classes),
            'dataset_type': dataset_type
        }
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return None

def analyze_all_models():
    """
    Analyze performance across all base models.
    
    Returns:
        dict: Performance data for all models
    """
    print("=" * 80)
    print("COMPREHENSIVE BASE MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    print("Analyzing coverage and MPIW for different base models across datasets")
    print("Methods: Standard Conformal Prediction (std_conf), CQR (cqr_conf), and Ensemble (ens_conf)")
    print("Datasets: COCO, BDD100K, and Cityscapes validation sets")
    print()
    
    results = {}
    
    for model_id, config in BASE_MODEL_CONFIGS.items():
        print(f"Processing {model_id}: {config['model_name']}")
        performance = load_model_performance(model_id, config)
        
        if performance is not None:
            results[model_id] = {
                'config': config,
                'performance': performance
            }
            print(f"  ✓ Successfully loaded {performance['n_trials']} trials")
        else:
            print(f"  ❌ Failed to load data for {model_id}")
        print()
    
    return results

def print_performance_summary(results):
    """
    Print a comprehensive performance summary with separate tables for each dataset.
    
    Args:
        results (dict): Performance data from analyze_all_models()
    """
    print("=" * 80)
    print("PERFORMANCE SUMMARY BY DATASET")
    print("=" * 80)
    
    if not results:
        print("❌ No data available for analysis!")
        return
    
    # Group results by dataset
    datasets = {'coco': [], 'bdd100k': [], 'cityscapes': []}
    
    for model_id, data in results.items():
        dataset_type = data['performance']['dataset_type']
        datasets[dataset_type].append((model_id, data))
    
    # Print tables for each dataset
    for dataset_name, models in datasets.items():
        if not models:
            continue
            
        print(f"\n{dataset_name.upper()} DATASET RESULTS:")
        print("=" * 100)
        print(f"{'Model':<35} {'Coverage':<35} {'MPIW':<30}")
        print("-" * 100)
        
        # Sort models by performance (coverage close to 90%, then by MPIW)
        sorted_models = []
        for model_id, data in models:
            perf = data['performance']
            coverage_diff = abs(perf['coverage_mean'] - 0.9)  # Distance from target 90%
            sorted_models.append((model_id, data, coverage_diff, perf['mpiw_mean']))
        
        # Sort by coverage difference first, then by MPIW
        sorted_models.sort(key=lambda x: (x[2], x[3]))
        
        # Print results for this dataset
        for model_id, data, _, _ in sorted_models:
            config = data['config']
            perf = data['performance']
            
            coverage_str = f"{perf['coverage_mean']:.3f} ± {perf['coverage_std']:.3f} (range: {perf['coverage_min']:.3f} - {perf['coverage_max']:.3f})"
            mpiw_str = f"{perf['mpiw_mean']:.1f} ± {perf['mpiw_std']:.1f} (range: {perf['mpiw_min']:.1f} - {perf['mpiw_max']:.1f})"
            
            print(f"{config['model_name']:<35} {coverage_str:<35} {mpiw_str:<30}")
        
        print("-" * 100)

def print_mpiw_at_90_coverage(results):
    """
    Print MPIW values at exactly 90% coverage (89-91% range) for each dataset.
    
    Args:
        results (dict): Performance data from analyze_all_models()
    """
    print("\n" + "=" * 80)
    print("MPIW AT 90% COVERAGE (ESTIMATED FROM 85-95% RANGE) BY DATASET")
    print("=" * 80)
    
    if not results:
        print("❌ No data available for analysis!")
        return
    
    # Group results by dataset
    datasets = {'coco': [], 'bdd100k': [], 'cityscapes': []}
    
    for model_id, data in results.items():
        dataset_type = data['performance']['dataset_type']
        datasets[dataset_type].append((model_id, data))
    
    # Print tables for each dataset showing MPIW at 90% coverage
    for dataset_name, models in datasets.items():
        if not models:
            continue
            
        print(f"\n{dataset_name.upper()} DATASET - MPIW AT 90% COVERAGE:")
        print("=" * 80)
        print(f"{'Model':<45} {'MPIW@90%':<15} {'Data Points':<12}")
        print("-" * 80)
        
        # Calculate MPIW at 90% coverage for each model
        models_at_90 = []
        for model_id, data in models:
            config = data['config']
            
            # Load the raw data again to get per-trial values
            file_path = os.path.join(config['directory'], config['data_file'])
            if not os.path.exists(file_path):
                continue
                
            try:
                control_data = torch.load(file_path, map_location='cpu', weights_only=False)
                
                # Determine valid classes based on dataset
                if 'cityscapes' in model_id.lower():
                    valid_classes = [0, 1, 2, 3, 5, 6, 7]
                elif 'bdd100k' in model_id.lower():
                    valid_classes = [0, 1, 2, 3, 5, 6, 7, 9]
                else:
                    valid_classes = list(range(control_data.shape[1]))
                
                # Extract per-trial coverage and MPIW
                coverage_by_class = control_data[:, :, :, METRIC_INDICES['cov_box']].mean(dim=2)
                mpiw_by_class = control_data[:, :, 0, METRIC_INDICES['mpiw']]
                
                # Filter trials with coverage between 89-91%
                valid_trial_mpiws = []
                for trial_idx in range(control_data.shape[0]):
                    valid_coverages = [coverage_by_class[trial_idx, cls].item() for cls in valid_classes if cls < control_data.shape[1]]
                    valid_mpiws = [mpiw_by_class[trial_idx, cls].item() for cls in valid_classes if cls < control_data.shape[1]]
                    
                    # Filter out zero values
                    valid_coverages = [c for c in valid_coverages if c > 0]
                    valid_mpiws = [m for m in valid_mpiws if m > 0]
                    
                    if valid_coverages and valid_mpiws:
                        trial_coverage = np.mean(valid_coverages)
                        trial_mpiw = np.mean(valid_mpiws)
                        
                        # Use wider range (85-95%) to get more data points for 90% estimation
                        if 0.85 <= trial_coverage <= 0.95:
                            valid_trial_mpiws.append((trial_coverage, trial_mpiw))
                
                if valid_trial_mpiws:
                    # Extract coverage and MPIW values
                    coverages = np.array([x[0] for x in valid_trial_mpiws])
                    mpiws = np.array([x[1] for x in valid_trial_mpiws])
                    
                    # Method 1: Find trials closest to 90% coverage
                    closest_to_90_indices = np.argsort(np.abs(coverages - 0.9))[:5]  # Take 5 closest
                    mpiw_at_90 = np.mean(mpiws[closest_to_90_indices])
                    
                    # Method 2: If we have enough data points, do linear interpolation
                    if len(valid_trial_mpiws) >= 3:
                        try:
                            from scipy.interpolate import interp1d
                            if len(np.unique(coverages)) >= 2:  # Need at least 2 unique coverage values
                                f = interp1d(coverages, mpiws, kind='linear', fill_value='extrapolate')
                                mpiw_at_90 = float(f(0.9))
                        except:
                            pass  # Fall back to closest method
                    
                    models_at_90.append((model_id, data, mpiw_at_90, len(valid_trial_mpiws)))
                    
            except Exception as e:
                continue
        
        # Sort by MPIW at 90% (lower is better)
        models_at_90.sort(key=lambda x: x[2])
        
        # Print results
        for model_id, data, mpiw_at_90, n_valid_trials in models_at_90:
            config = data['config']
            print(f"{config['model_name']:<45} {mpiw_at_90:<15.1f} {n_valid_trials:<12}")
        
        if not models_at_90:
            print("No models found with coverage in 85-95% range")
        
        print("-" * 80)

def find_best_models(results):
    """
    Identify the best performing models.
    
    Args:
        results (dict): Performance data from analyze_all_models()
    """
    if not results:
        return
        
    print("\n" + "=" * 80)
    print("BEST MODEL ANALYSIS")
    print("=" * 80)
    
    # Find model with coverage closest to 90%
    best_coverage_model = None
    best_coverage_diff = float('inf')
    
    # Find model with lowest MPIW (among those with reasonable coverage)
    best_mpiw_model = None
    best_mpiw_value = float('inf')
    
    for model_id, data in results.items():
        perf = data['performance']
        coverage_diff = abs(perf['coverage_mean'] - 0.9)
        
        # Best coverage
        if coverage_diff < best_coverage_diff:
            best_coverage_diff = coverage_diff
            best_coverage_model = (model_id, data)
        
        # Best MPIW (only consider models with coverage between 88-92%)
        if 0.88 <= perf['coverage_mean'] <= 0.92 and perf['mpiw_mean'] < best_mpiw_value:
            best_mpiw_value = perf['mpiw_mean']
            best_mpiw_model = (model_id, data)
    
    # Print best models
    if best_coverage_model:
        model_id, data = best_coverage_model
        perf = data['performance']
        print(f"🏆 BEST COVERAGE: {data['config']['model_name']} ({model_id})")
        print(f"   Coverage: {perf['coverage_mean']:.3f} (difference from 90%: {best_coverage_diff:.3f})")
        print(f"   MPIW: {perf['mpiw_mean']:.1f}")
    
    if best_mpiw_model and best_mpiw_model != best_coverage_model:
        model_id, data = best_mpiw_model
        perf = data['performance']
        print(f"\n🏆 BEST MPIW: {data['config']['model_name']} ({model_id})")
        print(f"   MPIW: {perf['mpiw_mean']:.1f}")
        print(f"   Coverage: {perf['coverage_mean']:.3f}")
    
    if best_mpiw_model == best_coverage_model:
        print(f"\n🏆 OVERALL BEST: {best_coverage_model[1]['config']['model_name']} "
              f"({best_coverage_model[0]}) - Best in both coverage and MPIW!")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to analyze all base models"""
    # Analyze all models
    results = analyze_all_models()
    
    # Print comprehensive summary
    print_performance_summary(results)
    
    # Print MPIW at 90% coverage tables
    print_mpiw_at_90_coverage(results)
    
    # Find and highlight best models
    find_best_models(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Key Insights:")
    print("• Lower MPIW = tighter prediction intervals (better)")
    print("• Coverage should be close to 90% (target)")
    print("• All models use standard conformal prediction method")
    print("• Results based on COCO, BDD100K, and Cityscapes validation sets")
    print("=" * 80)

if __name__ == "__main__":
    main()