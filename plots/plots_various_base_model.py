#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various Base Model Performance Analysis Script

This script analyzes coverage and MPIW performance across 5 different base models
using standard conformal prediction method.

Base Models Analyzed:
1. ResNet-50-C4 (No FPN)
2. ResNet-50-FPN  
3. ResNet-101-C4 (No FPN)
4. ResNet-101-FPN
5. ResNeXt-101-FPN

Output: Simple coverage and MPIW statistics for each base model
"""

import os
import torch
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base model directories and configurations
BASE_MODEL_CONFIGS = {
    'r50c4': {
        'model_name': 'ResNet-50-C4 (No FPN)',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_r50c4_std_rank_class',
        'data_file': 'std_conf_r50c4_std_rank_class_box_set.pt'
    },
    'r50fpn': {
        'model_name': 'ResNet-50-FPN',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_r50fpn_std_rank_class',
        'data_file': 'std_conf_r50fpn_std_rank_class_box_set.pt'
    },
    'r101c4': {
        'model_name': 'ResNet-101-C4 (No FPN)',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_r101c4_std_rank_class',
        'data_file': 'std_conf_r101c4_std_rank_class_box_set.pt'
    },
    'r101fpn': {
        'model_name': 'ResNet-101-FPN',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_r101fpn_std_rank_class',
        'data_file': 'std_conf_r101fpn_std_rank_class_box_set.pt'
    },
    'x101fpn': {
        'model_name': 'ResNeXt-101-FPN',
        'directory': '/ssd_4TB/divake/conformal-od/output/coco_val/std_conf_x101fpn_std_rank_class',
        'data_file': 'std_conf_x101fpn_std_rank_class_box_set.pt'
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
        
        # Extract metrics - average over classes and score indices for each trial
        coverage_all = control_data[:, :, :, METRIC_INDICES['cov_box']].mean(dim=(1,2))
        mpiw_all = control_data[:, :, 0, METRIC_INDICES['mpiw']].mean(dim=1)  # Use first score index only
        
        # Convert to numpy
        coverage_data = coverage_all.cpu().numpy()
        mpiw_data = mpiw_all.cpu().numpy()
        
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
            'n_classes': control_data.shape[1]
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
    print("VARIOUS BASE MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    print("Analyzing coverage and MPIW for 5 different base models")
    print("Method: Standard Conformal Prediction (std_conf)")
    print("Dataset: COCO validation set")
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
    Print a comprehensive performance summary.
    
    Args:
        results (dict): Performance data from analyze_all_models()
    """
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    if not results:
        print("❌ No data available for analysis!")
        return
    
    # Header
    print(f"{'Model':<25} {'Coverage':<15} {'MPIW':<15} {'Trials':<8}")
    print("-" * 80)
    
    # Sort models by performance (coverage close to 90%, then by MPIW)
    sorted_models = []
    for model_id, data in results.items():
        perf = data['performance']
        coverage_diff = abs(perf['coverage_mean'] - 0.9)  # Distance from target 90%
        sorted_models.append((model_id, data, coverage_diff, perf['mpiw_mean']))
    
    # Sort by coverage difference first, then by MPIW
    sorted_models.sort(key=lambda x: (x[2], x[3]))
    
    # Print results
    for model_id, data, _, _ in sorted_models:
        config = data['config']
        perf = data['performance']
        
        coverage_str = f"{perf['coverage_mean']:.3f} ± {perf['coverage_std']:.3f}"
        mpiw_str = f"{perf['mpiw_mean']:.1f} ± {perf['mpiw_std']:.1f}"
        
        print(f"{config['model_name']:<25} {coverage_str:<15} {mpiw_str:<15} {perf['n_trials']:<8}")
    
    print("-" * 80)
    
    # Detailed statistics
    print("\nDETAILED STATISTICS:")
    print("=" * 80)
    
    for model_id, data, _, _ in sorted_models:
        config = data['config']
        perf = data['performance']
        
        print(f"\n{config['model_name']} ({model_id}):")
        print(f"  Coverage: {perf['coverage_mean']:.3f} ± {perf['coverage_std']:.3f} "
              f"(range: {perf['coverage_min']:.3f} - {perf['coverage_max']:.3f})")
        print(f"  MPIW:     {perf['mpiw_mean']:.1f} ± {perf['mpiw_std']:.1f} "
              f"(range: {perf['mpiw_min']:.1f} - {perf['mpiw_max']:.1f})")
        print(f"  Trials:   {perf['n_trials']} calibration trials")
        print(f"  Classes:  {perf['n_classes']} object classes")

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
    
    # Find and highlight best models
    find_best_models(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Key Insights:")
    print("• Lower MPIW = tighter prediction intervals (better)")
    print("• Coverage should be close to 90% (target)")
    print("• All models use standard conformal prediction method")
    print("• Results based on COCO validation set")
    print("=" * 80)

if __name__ == "__main__":
    main()