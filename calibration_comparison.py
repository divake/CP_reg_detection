import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

def load_results(base_path="output/coco_val"):
    """Load results for both std and learn methods"""
    results = {}
    
    # Define paths
    std_path = os.path.join(base_path, "std_conf_x101fpn_std_rank_class")
    learn_path = os.path.join(base_path, "learn_conf_x101fpn_learn_rank_class")
    
    # Load CSV files - use learnable scores for learn method, absolute residual for std method
    results['std'] = pd.read_csv(os.path.join(std_path, "std_conf_x101fpn_std_rank_class_res_table_abs_res.csv"))
    results['learn'] = pd.read_csv(os.path.join(learn_path, "learn_conf_x101fpn_learn_rank_class_res_table_learn_res.csv"))
    
    # Load label tables
    results['std_label'] = pd.read_csv(os.path.join(std_path, "std_conf_x101fpn_std_rank_class_label_table.csv"))
    results['learn_label'] = pd.read_csv(os.path.join(learn_path, "learn_conf_x101fpn_learn_rank_class_label_table.csv"))
    
    return results

def calculate_calibration_metrics(results):
    """Calculate key calibration metrics for comparison"""
    metrics = {}
    
    for method in ['std', 'learn']:
        df = results[method]
        
        # Coverage metrics (should be close to 0.95 for well-calibrated)
        coverage_cols = ['cov x0', 'cov y0', 'cov x1', 'cov y1', 'cov box']
        
        # Get mean class rows for different sample sizes
        mean_all = df[df['class'] == 'mean class (nr calib > 0)'].iloc[0]
        mean_100 = df[df['class'] == 'mean class (nr calib >= 100)'].iloc[0]
        mean_1000 = df[df['class'] == 'mean class (nr calib >= 1000)'].iloc[0]
        
        metrics[method] = {
            'coverage': {
                'all': {col: mean_all[col] for col in coverage_cols},
                '100+': {col: mean_100[col] for col in coverage_cols},
                '1000+': {col: mean_1000[col] for col in coverage_cols}
            },
            'interval_width': {
                'all': mean_all['mpiw'],
                '100+': mean_100['mpiw'],
                '1000+': mean_1000['mpiw']
            },
            'box_stretch': {
                'all': mean_all['box stretch'],
                '100+': mean_100['box stretch'],
                '1000+': mean_1000['box stretch']
            },
            'quantiles': {
                'q_x0': mean_all['q x0'],
                'q_y0': mean_all['q y0'],
                'q_x1': mean_all['q x1'],
                'q_y1': mean_all['q y1']
            }
        }
    
    return metrics

def plot_coverage_comparison(metrics, save_path="plots/calibration_comparison"):
    """Create coverage comparison plots"""
    os.makedirs(save_path, exist_ok=True)
    
    # Figure 1: Coverage by coordinate and sample size
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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
        
        bars1 = ax.bar(x - width/2, std_coverage, width, label='Standard', alpha=0.8)
        bars2 = ax.bar(x + width/2, learn_coverage, width, label='Learnable', alpha=0.8)
        
        # Add nominal coverage line
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Nominal (0.95)')
        
        ax.set_xlabel('Coordinate')
        ax.set_ylabel('Coverage')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(coord_labels)
        ax.set_ylim(0.8, 1.0)
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
    
    # Figure 2: Calibration error analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    
    bars1 = ax.bar(x - width/2, std_errors, width, label='Standard', alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x + width/2, learn_errors, width, label='Learnable', alpha=0.8, color='#4ECDC4')
    
    ax.set_xlabel('Coordinate')
    ax.set_ylabel('Absolute Calibration Error')
    ax.set_title('Calibration Error: |Coverage - 0.95| (Classes with ≥1000 samples)')
    ax.set_xticks(x)
    ax.set_xticklabels(coords)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'calibration_error.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_interval_width_comparison(metrics, save_path="plots/calibration_comparison"):
    """Compare prediction interval widths"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Interval width comparison
    sample_sizes = ['all', '100+', '1000+']
    labels = ['All', '≥100', '≥1000']
    
    std_widths = [metrics['std']['interval_width'][size] for size in sample_sizes]
    learn_widths = [metrics['learn']['interval_width'][size] for size in sample_sizes]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, std_widths, width, label='Standard', alpha=0.8)
    bars2 = ax1.bar(x + width/2, learn_widths, width, label='Learnable', alpha=0.8)
    
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
    
    bars1 = ax2.bar(x - width/2, std_stretch, width, label='Standard', alpha=0.8)
    bars2 = ax2.bar(x + width/2, learn_stretch, width, label='Learnable', alpha=0.8)
    
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
    plt.savefig(os.path.join(save_path, 'interval_width_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_analysis(results, save_path="plots/calibration_comparison"):
    """Analyze calibration performance per class"""
    # Filter out summary rows
    std_df = results['std'][results['std']['class'].str.contains('^[a-z]', regex=True, na=False)]
    learn_df = results['learn'][results['learn']['class'].str.contains('^[a-z]', regex=True, na=False)]
    
    # Get classes with sufficient samples
    min_samples = 100
    std_filtered = std_df[std_df['nr calib'] >= min_samples]
    learn_filtered = learn_df[learn_df['nr calib'] >= min_samples]
    
    # Find common classes
    common_classes = set(std_filtered['class']) & set(learn_filtered['class'])
    common_classes = sorted(list(common_classes))
    
    # Create scatter plot comparing box coverage
    fig, ax = plt.subplots(figsize=(10, 8))
    
    std_box_cov = []
    learn_box_cov = []
    
    for cls in common_classes:
        std_cov = std_filtered[std_filtered['class'] == cls]['cov box'].values[0]
        learn_cov = learn_filtered[learn_filtered['class'] == cls]['cov box'].values[0]
        std_box_cov.append(std_cov)
        learn_box_cov.append(learn_cov)
    
    # Create scatter plot
    scatter = ax.scatter(std_box_cov, learn_box_cov, s=100, alpha=0.6)
    
    # Add diagonal line
    ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', alpha=0.5)
    
    # Add nominal coverage lines
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.3, label='Nominal (0.95)')
    ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.3)
    
    # Annotate some interesting points
    for i, cls in enumerate(common_classes):
        if abs(std_box_cov[i] - learn_box_cov[i]) > 0.02:  # Large differences
            ax.annotate(cls, (std_box_cov[i], learn_box_cov[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Standard Method Box Coverage')
    ax.set_ylabel('Learnable Method Box Coverage')
    ax.set_title('Per-Class Box Coverage Comparison (Classes with ≥100 samples)')
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'per_class_coverage_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(metrics, results, save_path="plots/calibration_comparison"):
    """Create a comprehensive dashboard summarizing all calibration metrics"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Calibration Performance Comparison: Standard vs Learnable Scoring Functions', 
                 fontsize=16, fontweight='bold')
    
    # 1. Coverage Summary (Top Left)
    ax1 = fig.add_subplot(gs[0, :2])
    
    methods = ['Standard', 'Learnable']
    metrics_names = ['Box Coverage', 'Mean Coord Coverage', 'MPIW', 'Box Stretch']
    
    std_values = [
        metrics['std']['coverage']['1000+']['cov box'],
        np.mean([metrics['std']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']]),
        metrics['std']['interval_width']['1000+']/100,  # Normalize for visualization
        metrics['std']['box_stretch']['1000+']
    ]
    
    learn_values = [
        metrics['learn']['coverage']['1000+']['cov box'],
        np.mean([metrics['learn']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']]),
        metrics['learn']['interval_width']['1000+']/100,  # Normalize for visualization
        metrics['learn']['box_stretch']['1000+']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, std_values, width, label='Standard', alpha=0.8)
    bars2 = ax1.bar(x + width/2, learn_values, width, label='Learnable', alpha=0.8)
    
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Value')
    ax1.set_title('Key Metrics Summary (Classes with ≥1000 samples)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Calibration Error Heatmap (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    coords = ['x₀', 'y₀', 'x₁', 'y₁', 'box']
    methods = ['Std', 'Learn']
    
    error_matrix = []
    for method in ['std', 'learn']:
        errors = []
        for coord in ['cov x0', 'cov y0', 'cov x1', 'cov y1', 'cov box']:
            errors.append(abs(metrics[method]['coverage']['1000+'][coord] - 0.95))
        error_matrix.append(errors)
    
    im = ax2.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.05)
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
    
    plt.colorbar(im, ax=ax2)
    
    # 3. Score Distribution (Middle Row)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Quantile comparison
    quantiles = ['q_x0', 'q_y0', 'q_x1', 'q_y1']
    std_quantiles = [metrics['std']['quantiles'][q] for q in quantiles]
    learn_quantiles = [metrics['learn']['quantiles'][q] for q in quantiles]
    
    x = np.arange(len(quantiles))
    bars1 = ax3.bar(x - width/2, std_quantiles, width, label='Standard', alpha=0.8)
    bars2 = ax3.bar(x + width/2, learn_quantiles, width, label='Learnable', alpha=0.8)
    
    ax3.set_xlabel('Quantile')
    ax3.set_ylabel('Value')
    ax3.set_title('Quantile Values Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['x₀', 'y₀', 'x₁', 'y₁'])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Summary Statistics Table (Bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Standard', 'Learnable', 'Difference', 'Better'],
        ['Box Coverage (1000+)', 
         f"{metrics['std']['coverage']['1000+']['cov box']:.4f}",
         f"{metrics['learn']['coverage']['1000+']['cov box']:.4f}",
         f"{abs(metrics['std']['coverage']['1000+']['cov box'] - metrics['learn']['coverage']['1000+']['cov box']):.4f}",
         'Learnable' if abs(metrics['learn']['coverage']['1000+']['cov box'] - 0.95) < abs(metrics['std']['coverage']['1000+']['cov box'] - 0.95) else 'Standard'],
        ['Mean Coord Coverage (1000+)', 
         f"{np.mean([metrics['std']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']]):.4f}",
         f"{np.mean([metrics['learn']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']]):.4f}",
         f"{abs(np.mean([metrics['std']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']]) - np.mean([metrics['learn']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']])):.4f}",
         'Learnable' if abs(np.mean([metrics['learn']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']]) - 0.95) < abs(np.mean([metrics['std']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']]) - 0.95) else 'Standard'],
        ['MPIW (1000+)', 
         f"{metrics['std']['interval_width']['1000+']:.2f}",
         f"{metrics['learn']['interval_width']['1000+']:.2f}",
         f"{metrics['std']['interval_width']['1000+'] - metrics['learn']['interval_width']['1000+']:.2f}",
         'Learnable' if metrics['learn']['interval_width']['1000+'] < metrics['std']['interval_width']['1000+'] else 'Standard'],
        ['Box Stretch (1000+)', 
         f"{metrics['std']['box_stretch']['1000+']:.4f}",
         f"{metrics['learn']['box_stretch']['1000+']:.4f}",
         f"{abs(metrics['std']['box_stretch']['1000+'] - metrics['learn']['box_stretch']['1000+']):.4f}",
         'Learnable' if metrics['learn']['box_stretch']['1000+'] < metrics['std']['box_stretch']['1000+'] else 'Standard']
    ]
    
    table = ax4.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the "Better" column
    for i in range(1, 5):
        if summary_data[i][4] == 'Learnable':
            table[(i, 4)].set_facecolor('#4ECDC4')
        else:
            table[(i, 4)].set_facecolor('#FF6B6B')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'calibration_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the complete comparison"""
    print("Loading results...")
    results = load_results()
    
    print("Calculating calibration metrics...")
    metrics = calculate_calibration_metrics(results)
    
    print("Creating visualizations...")
    save_path = "plots/calibration_comparison"
    os.makedirs(save_path, exist_ok=True)
    
    # Generate all plots
    plot_coverage_comparison(metrics, save_path)
    plot_interval_width_comparison(metrics, save_path)
    plot_per_class_analysis(results, save_path)
    create_summary_dashboard(metrics, results, save_path)
    
    # Print summary to console
    print("\n" + "="*60)
    print("CALIBRATION COMPARISON SUMMARY")
    print("="*60)
    
    print("\nFor classes with ≥1000 calibration samples:")
    print("\nBox Coverage (Target: 0.95):")
    print(f"  Standard: {metrics['std']['coverage']['1000+']['cov box']:.4f}")
    print(f"  Learnable: {metrics['learn']['coverage']['1000+']['cov box']:.4f}")
    
    print("\nMean Coordinate Coverage:")
    std_mean_cov = np.mean([metrics['std']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']])
    learn_mean_cov = np.mean([metrics['learn']['coverage']['1000+'][f'cov {coord}'] for coord in ['x0', 'y0', 'x1', 'y1']])
    print(f"  Standard: {std_mean_cov:.4f}")
    print(f"  Learnable: {learn_mean_cov:.4f}")
    
    print("\nMean Prediction Interval Width (MPIW):")
    print(f"  Standard: {metrics['std']['interval_width']['1000+']:.2f}")
    print(f"  Learnable: {metrics['learn']['interval_width']['1000+']:.2f}")
    
    print("\nBox Stretch Factor:")
    print(f"  Standard: {metrics['std']['box_stretch']['1000+']:.4f}")
    print(f"  Learnable: {metrics['learn']['box_stretch']['1000+']:.4f}")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    
    # Determine which method is better calibrated
    std_box_error = abs(metrics['std']['coverage']['1000+']['cov box'] - 0.95)
    learn_box_error = abs(metrics['learn']['coverage']['1000+']['cov box'] - 0.95)
    
    if learn_box_error < std_box_error:
        print("✓ Learnable method shows BETTER calibration (closer to nominal coverage)")
    else:
        print("✓ Standard method shows BETTER calibration (closer to nominal coverage)")
    
    if metrics['learn']['interval_width']['1000+'] < metrics['std']['interval_width']['1000+']:
        print("✓ Learnable method produces NARROWER prediction intervals")
    else:
        print("✓ Standard method produces NARROWER prediction intervals")
    
    print("\nPlots saved to:", save_path)
    print("="*60)

if __name__ == "__main__":
    main() 