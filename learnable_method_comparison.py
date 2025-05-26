#!/usr/bin/env python3
"""
Comprehensive comparison script for the new learnable conformal scoring function.

This script compares our new learnable conformal method against the standard conformal method
and provides detailed analysis of the performance improvements.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')

sns.set_palette("husl")


def load_comparison_results(base_path="output/coco_val"):
    """Load results for both standard and learnable methods"""
    results = {}
    
    # Define paths
    std_path = os.path.join(base_path, "std_conf_x101fpn_std_rank_class")
    learn_path = os.path.join(base_path, "learn_conf_x101fpn")
    
    # Load CSV files - using appropriate scores for each method
    results['std'] = pd.read_csv(os.path.join(std_path, "std_conf_x101fpn_std_rank_class_res_table_abs_res.csv"))
    results['learn'] = pd.read_csv(os.path.join(learn_path, "learn_conf_x101fpn_res_table_learn_res.csv"))
    
    # Load label tables
    results['std_label'] = pd.read_csv(os.path.join(std_path, "std_conf_x101fpn_std_rank_class_label_table.csv"))
    results['learn_label'] = pd.read_csv(os.path.join(learn_path, "learn_conf_x101fpn_label_table.csv"))
    
    return results


def calculate_comparison_metrics(results):
    """Calculate comprehensive comparison metrics"""
    metrics = {}
    
    # Define sample size categories
    categories = {
        'all': 0,
        '100+': 1,
        '1000+': 2
    }
    
    for method in ['std', 'learn']:
        metrics[method] = {
            'coverage': {},
            'interval_width': {},
            'box_stretch': {},
            'quantiles': {}
        }
        
        df = results[method]
        
        # Extract metrics for different sample size categories
        for cat_name, row_idx in categories.items():
            row = df.iloc[row_idx]
            
            metrics[method]['coverage'][cat_name] = {
                'cov x0': row['cov x0'],
                'cov y0': row['cov y0'],
                'cov x1': row['cov x1'],
                'cov y1': row['cov y1'],
                'cov box': row['cov box']
            }
            
            metrics[method]['interval_width'][cat_name] = row['mpiw']
            metrics[method]['box_stretch'][cat_name] = row['box stretch']
            
            if cat_name == '1000+':  # Store quantiles for the most reliable category
                metrics[method]['quantiles'] = {
                    'q_x0': row['q x0'],
                    'q_y0': row['q y0'],
                    'q_x1': row['q x1'],
                    'q_y1': row['q y1']
                }
    
    return metrics


def plot_coverage_comparison(metrics, save_path="plots/learnable_method_comparison"):
    """Create detailed coverage comparison plots"""
    os.makedirs(save_path, exist_ok=True)
    
    # Figure 1: Coverage by coordinate and sample size
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sample_sizes = ['all', '100+', '1000+']
    titles = ['All Classes', 'Classes with ≥100 samples', 'Classes with ≥1000 samples']
    
    for idx, (size, title) in enumerate(zip(sample_sizes, titles)):
        ax = axes[idx]
        
        coords = ['cov x0', 'cov y0', 'cov x1', 'cov y1', 'cov box']
        coord_labels = ['x₀', 'y₀', 'x₁', 'y₁', 'box']
        
        std_coverage = [metrics['std']['coverage'][size][coord] for coord in coords]
        learn_coverage = [metrics['learn']['coverage'][size][coord] for coord in coords]
        
        x = np.arange(len(coord_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, std_coverage, width, label='Standard', alpha=0.8, color='#FF6B6B')
        bars2 = ax.bar(x + width/2, learn_coverage, width, label='Learnable', alpha=0.8, color='#4ECDC4')
        
        # Add nominal coverage line
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (0.95)')
        
        ax.set_xlabel('Coordinate')
        ax.set_ylabel('Coverage')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(coord_labels)
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'coverage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_analysis(metrics, save_path="plots/learnable_method_comparison"):
    """Create calibration error analysis plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calibration error analysis for 1000+ samples (most reliable)
    nominal = 0.95
    coords = ['x₀', 'y₀', 'x₁', 'y₁', 'box']
    
    # Calculate calibration errors (deviation from nominal)
    std_errors = []
    learn_errors = []
    
    for coord in ['cov x0', 'cov y0', 'cov x1', 'cov y1', 'cov box']:
        std_errors.append(abs(metrics['std']['coverage']['1000+'][coord] - nominal))
        learn_errors.append(abs(metrics['learn']['coverage']['1000+'][coord] - nominal))
    
    x = np.arange(len(coords))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, std_errors, width, label='Standard', alpha=0.8, color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, learn_errors, width, label='Learnable', alpha=0.8, color='#4ECDC4')
    
    ax1.set_xlabel('Coordinate')
    ax1.set_ylabel('Absolute Calibration Error')
    ax1.set_title('Calibration Error: |Coverage - 0.95| (Classes with ≥1000 samples)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(coords)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    # Quantile comparison
    quantiles = ['q_x0', 'q_y0', 'q_x1', 'q_y1']
    std_quantiles = [metrics['std']['quantiles'][q] for q in quantiles]
    learn_quantiles = [metrics['learn']['quantiles'][q] for q in quantiles]
    
    x = np.arange(len(quantiles))
    bars1 = ax2.bar(x - width/2, std_quantiles, width, label='Standard', alpha=0.8, color='#FF6B6B')
    bars2 = ax2.bar(x + width/2, learn_quantiles, width, label='Learnable', alpha=0.8, color='#4ECDC4')
    
    ax2.set_xlabel('Quantile')
    ax2.set_ylabel('Value')
    ax2.set_title('Quantile Values Comparison (Classes with ≥1000 samples)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['x₀', 'y₀', 'x₁', 'y₁'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'calibration_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_efficiency_comparison(metrics, save_path="plots/learnable_method_comparison"):
    """Compare prediction interval widths and efficiency"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Interval width comparison
    sample_sizes = ['all', '100+', '1000+']
    labels = ['All', '≥100', '≥1000']
    
    std_widths = [metrics['std']['interval_width'][size] for size in sample_sizes]
    learn_widths = [metrics['learn']['interval_width'][size] for size in sample_sizes]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, std_widths, width, label='Standard', alpha=0.8, color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, learn_widths, width, label='Learnable', alpha=0.8, color='#4ECDC4')
    
    ax1.set_xlabel('Sample Size Category')
    ax1.set_ylabel('Mean Prediction Interval Width (MPIW)')
    ax1.set_title('Prediction Interval Width Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Box stretch comparison
    std_stretch = [metrics['std']['box_stretch'][size] for size in sample_sizes]
    learn_stretch = [metrics['learn']['box_stretch'][size] for size in sample_sizes]
    
    bars1 = ax2.bar(x - width/2, std_stretch, width, label='Standard', alpha=0.8, color='#FF6B6B')
    bars2 = ax2.bar(x + width/2, learn_stretch, width, label='Learnable', alpha=0.8, color='#4ECDC4')
    
    ax2.set_xlabel('Sample Size Category')
    ax2.set_ylabel('Box Stretch Factor')
    ax2.set_title('Box Stretch Factor Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_dashboard(metrics, save_path="plots/learnable_method_comparison"):
    """Create a comprehensive dashboard summarizing the comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Title
    fig.suptitle('Learnable vs Standard Conformal Prediction Methods: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. Coverage Summary (Top Left)
    ax1 = axes[0, 0]
    
    coords = ['x₀', 'y₀', 'x₁', 'y₁', 'box']
    std_coverage = [metrics['std']['coverage']['1000+'][f'cov {coord.replace("₀", "0").replace("₁", "1")}'] for coord in coords]
    learn_coverage = [metrics['learn']['coverage']['1000+'][f'cov {coord.replace("₀", "0").replace("₁", "1")}'] for coord in coords]
    
    x = np.arange(len(coords))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, std_coverage, width, label='Standard', alpha=0.8, color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, learn_coverage, width, label='Learnable', alpha=0.8, color='#4ECDC4')
    
    ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (0.95)')
    ax1.set_xlabel('Coordinate')
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage Comparison (Classes with ≥1000 samples)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(coords)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Calibration Error Heatmap (Top Right)
    ax2 = axes[0, 1]
    
    methods = ['Standard', 'Learnable']
    error_matrix = []
    
    for method in ['std', 'learn']:
        errors = []
        for coord in ['cov x0', 'cov y0', 'cov x1', 'cov y1', 'cov box']:
            errors.append(abs(metrics[method]['coverage']['1000+'][coord] - 0.95))
        error_matrix.append(errors)
    
    im = ax2.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=max(max(row) for row in error_matrix))
    ax2.set_xticks(np.arange(len(coords)))
    ax2.set_yticks(np.arange(len(methods)))
    ax2.set_xticklabels(coords)
    ax2.set_yticklabels(methods)
    ax2.set_title('Calibration Error Heatmap')
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(coords)):
            text = ax2.text(j, i, f'{error_matrix[i][j]:.3f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    # 3. Efficiency Metrics (Bottom Left)
    ax3 = axes[1, 0]
    
    metrics_names = ['MPIW', 'Box Stretch']
    std_values = [metrics['std']['interval_width']['1000+'], metrics['std']['box_stretch']['1000+']]
    learn_values = [metrics['learn']['interval_width']['1000+'], metrics['learn']['box_stretch']['1000+']]
    
    x = np.arange(len(metrics_names))
    bars1 = ax3.bar(x - width/2, std_values, width, label='Standard', alpha=0.8, color='#FF6B6B')
    bars2 = ax3.bar(x + width/2, learn_values, width, label='Learnable', alpha=0.8, color='#4ECDC4')
    
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Value')
    ax3.set_title('Efficiency Metrics (Classes with ≥1000 samples)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Performance Summary (Bottom Right) - Text Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    std_box_error = abs(metrics['std']['coverage']['1000+']['cov box'] - 0.95)
    learn_box_error = abs(metrics['learn']['coverage']['1000+']['cov box'] - 0.95)
    
    std_mean_coord_error = np.mean([abs(metrics['std']['coverage']['1000+'][f'cov {coord}'] - 0.95) 
                                   for coord in ['x0', 'y0', 'x1', 'y1']])
    learn_mean_coord_error = np.mean([abs(metrics['learn']['coverage']['1000+'][f'cov {coord}'] - 0.95) 
                                     for coord in ['x0', 'y0', 'x1', 'y1']])
    
    efficiency_improvement = ((metrics['std']['interval_width']['1000+'] - metrics['learn']['interval_width']['1000+']) / 
                            metrics['std']['interval_width']['1000+']) * 100
    
    summary_text = f"""PERFORMANCE SUMMARY (≥1000 samples):

Coverage Performance:
• Standard box coverage: {metrics['std']['coverage']['1000+']['cov box']:.3f}
• Learnable box coverage: {metrics['learn']['coverage']['1000+']['cov box']:.3f}
• Box calibration error improvement: {std_box_error - learn_box_error:.4f}

• Mean coordinate calibration error:
  - Standard: {std_mean_coord_error:.4f}
  - Learnable: {learn_mean_coord_error:.4f}

Efficiency:
• MPIW reduction: {efficiency_improvement:.1f}%
• Standard MPIW: {metrics['std']['interval_width']['1000+']:.1f}
• Learnable MPIW: {metrics['learn']['interval_width']['1000+']:.1f}

Key Insights:
{"✓ Better calibration" if learn_box_error < std_box_error else "✗ Worse calibration"}
{"✓ Narrower intervals" if efficiency_improvement > 0 else "✗ Wider intervals"}
{"✓ More efficient" if learn_mean_coord_error < std_mean_coord_error else "✗ Less efficient"}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comprehensive_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_detailed_summary(metrics):
    """Print detailed comparison summary to console"""
    print("\n" + "="*80)
    print("LEARNABLE vs STANDARD CONFORMAL PREDICTION - DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n📊 COVERAGE ANALYSIS (Classes with ≥1000 calibration samples):")
    print("-" * 60)
    
    coords = ['x0', 'y0', 'x1', 'y1', 'box']
    coord_names = ['X₀ (left)', 'Y₀ (top)', 'X₁ (right)', 'Y₁ (bottom)', 'Box (overall)']
    
    print(f"{'Coordinate':<15} {'Standard':<10} {'Learnable':<10} {'Difference':<12} {'Better':<10}")
    print("-" * 60)
    
    for coord, name in zip(coords, coord_names):
        std_cov = metrics['std']['coverage']['1000+'][f'cov {coord}']
        learn_cov = metrics['learn']['coverage']['1000+'][f'cov {coord}']
        diff = learn_cov - std_cov
        better = "Learnable" if abs(learn_cov - 0.95) < abs(std_cov - 0.95) else "Standard"
        
        print(f"{name:<15} {std_cov:<10.4f} {learn_cov:<10.4f} {diff:<+12.4f} {better:<10}")
    
    print(f"\n📏 EFFICIENCY ANALYSIS:")
    print("-" * 40)
    
    for size_cat in ['all', '100+', '1000+']:
        std_width = metrics['std']['interval_width'][size_cat]
        learn_width = metrics['learn']['interval_width'][size_cat]
        improvement = ((std_width - learn_width) / std_width) * 100
        
        print(f"MPIW ({size_cat:<5}): {std_width:>7.1f} → {learn_width:>7.1f} ({improvement:+.1f}%)")
    
    print(f"\n🎯 CALIBRATION QUALITY:")
    print("-" * 30)
    
    # Overall calibration error
    std_box_error = abs(metrics['std']['coverage']['1000+']['cov box'] - 0.95)
    learn_box_error = abs(metrics['learn']['coverage']['1000+']['cov box'] - 0.95)
    
    print(f"Box calibration error:")
    print(f"  Standard:  {std_box_error:.4f}")
    print(f"  Learnable: {learn_box_error:.4f}")
    print(f"  Improvement: {std_box_error - learn_box_error:+.4f}")
    
    # Mean coordinate calibration error
    std_coord_errors = [abs(metrics['std']['coverage']['1000+'][f'cov {coord}'] - 0.95) 
                       for coord in ['x0', 'y0', 'x1', 'y1']]
    learn_coord_errors = [abs(metrics['learn']['coverage']['1000+'][f'cov {coord}'] - 0.95) 
                         for coord in ['x0', 'y0', 'x1', 'y1']]
    
    std_mean_error = np.mean(std_coord_errors)
    learn_mean_error = np.mean(learn_coord_errors)
    
    print(f"\nMean coordinate calibration error:")
    print(f"  Standard:  {std_mean_error:.4f}")
    print(f"  Learnable: {learn_mean_error:.4f}")
    print(f"  Improvement: {std_mean_error - learn_mean_error:+.4f}")
    
    print(f"\n📈 QUANTILE VALUES (1000+ samples):")
    print("-" * 40)
    
    for coord in ['x0', 'y0', 'x1', 'y1']:
        std_q = metrics['std']['quantiles'][f'q_{coord}']
        learn_q = metrics['learn']['quantiles'][f'q_{coord}']
        ratio = learn_q / std_q if std_q > 0 else float('inf')
        
        print(f"q_{coord}: {std_q:>8.2f} → {learn_q:>8.2f} (ratio: {ratio:.3f})")
    
    print(f"\n🏆 FINAL VERDICT:")
    print("="*40)
    
    # Determine overall winner
    calibration_wins = sum(1 for coord in ['x0', 'y0', 'x1', 'y1', 'box']
                          if abs(metrics['learn']['coverage']['1000+'][f'cov {coord}'] - 0.95) < 
                             abs(metrics['std']['coverage']['1000+'][f'cov {coord}'] - 0.95))
    
    efficiency_improvement = ((metrics['std']['interval_width']['1000+'] - 
                              metrics['learn']['interval_width']['1000+']) / 
                             metrics['std']['interval_width']['1000+']) * 100
    
    print(f"✓ Calibration: Learnable wins {calibration_wins}/5 coordinates")
    print(f"✓ Efficiency: {efficiency_improvement:+.1f}% interval width change")
    
    if calibration_wins >= 3 and efficiency_improvement > 0:
        print("🎉 LEARNABLE METHOD CLEARLY SUPERIOR!")
    elif calibration_wins >= 3:
        print("👍 LEARNABLE METHOD SHOWS BETTER CALIBRATION")
    elif efficiency_improvement > 0:
        print("⚡ LEARNABLE METHOD MORE EFFICIENT")
    else:
        print("🤔 MIXED RESULTS - FURTHER ANALYSIS NEEDED")
    
    print("="*80)


def main():
    """Main function to run the comprehensive comparison"""
    print("🚀 Starting comprehensive comparison of learnable vs standard conformal methods...")
    
    # Load results
    print("📂 Loading results...")
    results = load_comparison_results()
    
    # Calculate metrics
    print("📊 Calculating comparison metrics...")
    metrics = calculate_comparison_metrics(results)
    
    # Create output directory
    save_path = "plots/learnable_method_comparison"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    print("🎨 Creating visualizations...")
    plot_coverage_comparison(metrics, save_path)
    plot_calibration_analysis(metrics, save_path)
    plot_efficiency_comparison(metrics, save_path)
    create_summary_dashboard(metrics, save_path)
    
    # Print detailed summary
    print_detailed_summary(metrics)
    
    # Save metrics to JSON for further analysis
    metrics_file = os.path.join(save_path, 'comparison_metrics.json')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Deep convert the metrics dictionary
    json_metrics = json.loads(json.dumps(metrics, default=convert_numpy))
    
    with open(metrics_file, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"\n📁 All plots saved to: {save_path}")
    print(f"📄 Metrics saved to: {metrics_file}")
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main() 