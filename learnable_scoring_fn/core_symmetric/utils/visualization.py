"""Visualization utilities for symmetric adaptive training with paper-ready plots and CSV export."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
})

# Define a professional color palette
COLORS = {
    'train': '#1f77b4',  # Blue
    'val': '#ff7f0e',    # Orange
    'coverage': '#2ca02c',  # Green
    'mpiw': '#d62728',     # Red
    'tau': '#9467bd',      # Purple
    'lr': '#8c564b',       # Brown
    'small': '#17becf',    # Cyan
    'medium': '#bcbd22',   # Olive
    'large': '#e377c2',    # Pink
    'target': '#7f7f7f',   # Gray
}


def save_training_csv(history: Dict[str, List], size_metrics: List[Dict], 
                     save_path: Path, model_name: str = "symmetric_adaptive"):
    """
    Save comprehensive training history to CSV file.
    
    Args:
        history: Training history dictionary
        size_metrics: List of size-stratified metrics per epoch
        save_path: Path to save CSV
        model_name: Model name for the CSV
    """
    # Prepare data for CSV
    csv_data = []
    
    # Get number of epochs
    n_epochs = max(len(v) for v in history.values() if isinstance(v, list))
    
    for epoch in range(n_epochs):
        row = {'epoch': epoch + 1, 'model': model_name}
        
        # Add all history metrics - handle both naming conventions
        for key, values in history.items():
            if isinstance(values, list) and epoch < len(values):
                # Handle different key names
                if key == 'coverage_rate':
                    row['val_coverage'] = values[epoch]
                elif key == 'avg_mpiw':
                    row['val_mpiw'] = values[epoch]
                elif key == 'total' and 'train_loss' not in history:
                    row['val_loss'] = values[epoch]
                else:
                    row[key] = values[epoch]
        
        # Add size-stratified metrics
        if epoch < len(size_metrics):
            size_data = size_metrics[epoch]
            for size_name in ['small', 'medium', 'large']:
                if size_name in size_data:
                    row[f'{size_name}_coverage'] = size_data[size_name]['coverage']
                    row[f'{size_name}_mpiw'] = size_data[size_name]['mpiw']
                    row[f'{size_name}_n'] = size_data[size_name].get('n', size_data[size_name].get('count', 0))
        
        csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    
    # Reorder columns for better readability
    column_order = ['epoch', 'model']
    
    # Core metrics
    core_metrics = ['train_loss', 'val_loss', 'val_coverage', 'val_mpiw', 'tau', 'learning_rate']
    column_order.extend([col for col in core_metrics if col in df.columns])
    
    # Size-stratified metrics
    for size in ['small', 'medium', 'large']:
        size_cols = [f'{size}_coverage', f'{size}_mpiw', f'{size}_n']
        column_order.extend([col for col in size_cols if col in df.columns])
    
    # Any remaining columns
    remaining_cols = [col for col in df.columns if col not in column_order]
    column_order.extend(remaining_cols)
    
    df = df[column_order]
    df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"📊 Saved training history to CSV: {save_path}")


def plot_training_results(
    history: Dict[str, List],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Create comprehensive training result plots with proper key handling.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Loss curves
    ax = axes[0, 0]
    train_loss = history.get('train_loss', [])
    # Handle different naming conventions for validation loss
    val_loss = history.get('val_loss', history.get('total', []))
    
    if train_loss and val_loss:
        epochs = range(1, min(len(train_loss), len(val_loss)) + 1)
        ax.plot(epochs, train_loss[:len(epochs)], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, val_loss[:len(epochs)], 'r-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Coverage evolution - handle both naming conventions
    ax = axes[0, 1]
    coverage = history.get('val_coverage', history.get('coverage_rate', []))
    
    if coverage:
        epochs = range(1, len(coverage) + 1)
        ax.plot(epochs, coverage, 'g-', linewidth=2)
        ax.axhline(y=0.89, color='r', linestyle='--', label='Target (89%)')
        ax.axhspan(0.88, 0.905, alpha=0.2, color='green', label='Target Zone')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Coverage Rate')
        ax.set_title('Validation Coverage Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.7, 1.0)
    
    # 3. MPIW evolution - handle both naming conventions
    ax = axes[1, 0]
    mpiw = history.get('val_mpiw', history.get('avg_mpiw', []))
    
    if mpiw:
        epochs = range(1, len(mpiw) + 1)
        ax.plot(epochs, mpiw, 'm-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MPIW (pixels)')
        ax.set_title('Mean Prediction Interval Width')
        ax.grid(True, alpha=0.3)
    
    # 4. Tau evolution
    ax = axes[1, 1]
    if 'tau' in history:
        epochs = range(1, len(history['tau']) + 1)
        ax.plot(epochs, history['tau'], 'c-', linewidth=2)
        ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Tau')
        ax.set_title('Calibration Factor (Tau) Evolution')
        ax.grid(True, alpha=0.3)
    
    # 5. Coverage vs MPIW scatter
    ax = axes[2, 0]
    # Use whichever keys are available
    coverage = history.get('val_coverage', history.get('coverage_rate', []))
    mpiw = history.get('val_mpiw', history.get('avg_mpiw', []))
    
    if coverage and mpiw and len(coverage) == len(mpiw):
        # Color by epoch
        scatter = ax.scatter(coverage, mpiw, c=range(len(coverage)), 
                           cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        # Mark start and end
        ax.scatter(coverage[0], mpiw[0], color='green', s=200, 
                  marker='s', label='Start', edgecolors='black', linewidth=2)
        ax.scatter(coverage[-1], mpiw[-1], color='red', s=200, 
                  marker='*', label='End', edgecolors='black', linewidth=2)
        
        ax.axvline(x=0.89, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Coverage Rate')
        ax.set_ylabel('MPIW (pixels)')
        ax.set_title('Coverage-Efficiency Tradeoff Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Epoch')
    
    # 6. Learning rate schedule
    ax = axes[2, 1]
    if 'learning_rate' in history:
        epochs = range(1, len(history['learning_rate']) + 1)
        ax.plot(epochs, history['learning_rate'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_tau_evolution(
    tau_history: List[float],
    coverage_history: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot detailed tau evolution.
    
    Args:
        tau_history: List of tau values
        coverage_history: Optional list of coverage values
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = range(len(tau_history))
    
    # Plot tau
    ax.plot(steps, tau_history, 'b-', linewidth=2, label='Tau')
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Tau=1.0')
    
    # Highlight regions
    tau_array = np.array(tau_history)
    ax.fill_between(steps, 1.0, tau_array, where=(tau_array > 1.0),
                    alpha=0.2, color='red', label='Over-correction')
    ax.fill_between(steps, tau_array, 1.0, where=(tau_array < 1.0),
                    alpha=0.2, color='blue', label='Under-correction')
    
    ax.set_xlabel('Calibration Step')
    ax.set_ylabel('Tau Value')
    ax.set_title('Tau Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add coverage on secondary axis if provided
    if coverage_history and len(coverage_history) == len(tau_history):
        ax2 = ax.twinx()
        ax2.plot(steps, coverage_history, 'g--', linewidth=1.5, alpha=0.7, label='Coverage')
        ax2.axhline(y=0.9, color='g', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Coverage Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0.7, 1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_width_distribution(
    widths: torch.Tensor,
    object_sizes: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot distribution of predicted widths.
    
    Args:
        widths: Predicted widths [N, 4]
        object_sizes: Optional object sizes for stratification
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    coord_names = ['x0', 'y0', 'x1', 'y1']
    
    for i, (ax, name) in enumerate(zip(axes.flat, coord_names)):
        width_data = widths[:, i].cpu().numpy()
        
        # Histogram
        ax.hist(width_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=width_data.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {width_data.mean():.1f}')
        ax.set_xlabel(f'Width for {name} (pixels)')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Widths - {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_size_stratified_results(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot size-stratified coverage and MPIW results.
    
    Args:
        results: Dictionary with size categories and their metrics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    categories = list(results.keys())
    coverages = [results[cat]['coverage'] for cat in categories]
    mpiws = [results[cat]['mpiw'] for cat in categories]
    
    # Coverage by size
    bars1 = ax1.bar(categories, coverages, color='skyblue', edgecolor='black')
    ax1.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    ax1.set_xlabel('Object Size Category')
    ax1.set_ylabel('Coverage Rate')
    ax1.set_title('Coverage by Object Size')
    ax1.legend()
    ax1.set_ylim(0.7, 1.0)
    
    # Add value labels on bars
    for bar, val in zip(bars1, coverages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # MPIW by size
    bars2 = ax2.bar(categories, mpiws, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Object Size Category')
    ax2.set_ylabel('Average MPIW (pixels)')
    ax2.set_title('MPIW by Object Size')
    
    # Add value labels on bars
    for bar, val in zip(bars2, mpiws):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comprehensive_training_results(
    history: Dict[str, List],
    save_path: Optional[Path] = None,
    show: bool = True,
    title_prefix: str = "Symmetric Adaptive Conformal Prediction"
):
    """
    Create comprehensive, paper-ready training result plots.
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Title
    fig.suptitle(f"{title_prefix} - Training Results", fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Loss curves (larger plot)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_loss_curves(ax1, history)
    
    # 2. Coverage evolution
    ax2 = fig.add_subplot(gs[0, 2])
    plot_coverage_evolution(ax2, history)
    
    # 3. MPIW evolution
    ax3 = fig.add_subplot(gs[1, 0])
    plot_mpiw_evolution(ax3, history)
    
    # 4. Tau evolution
    ax4 = fig.add_subplot(gs[1, 1])
    plot_tau_evolution_subplot(ax4, history)
    
    # 5. Coverage vs MPIW tradeoff
    ax5 = fig.add_subplot(gs[1, 2])
    plot_coverage_mpiw_tradeoff(ax5, history)
    
    # 6. Size-stratified coverage over time
    ax6 = fig.add_subplot(gs[2, :])
    plot_size_stratified_evolution(ax6, history, metric='coverage')
    
    # 7. Size-stratified MPIW over time
    ax7 = fig.add_subplot(gs[3, :])
    plot_size_stratified_evolution(ax7, history, metric='mpiw')
    
    # Add timestamp and metadata
    coverage = history.get('coverage_rate', history.get('val_coverage', []))
    timestamp_text = f"Training completed with {len(coverage)} epochs"
    fig.text(0.99, 0.01, timestamp_text, ha='right', va='bottom', fontsize=10, alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Saved comprehensive training plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_curves(ax, history):
    """Plot training and validation loss curves."""
    train_loss = history.get('train_loss', [])
    val_loss = history.get('total', history.get('val_loss', []))
    
    if train_loss and val_loss:
        epochs = range(1, min(len(train_loss), len(val_loss)) + 1)
        ax.plot(epochs, train_loss[:len(epochs)], color=COLORS['train'], label='Training Loss', 
                linewidth=2.5, marker='o', markersize=4, markevery=5)
        ax.plot(epochs, val_loss[:len(epochs)], color=COLORS['val'], label='Validation Loss', 
                linewidth=2.5, marker='s', markersize=4, markevery=5)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Training and Validation Loss', fontweight='bold', pad=10)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')


def plot_coverage_evolution(ax, history):
    """Plot coverage evolution with target zone."""
    coverage = history.get('coverage_rate', history.get('val_coverage', []))
    
    if coverage:
        epochs = range(1, len(coverage) + 1)
        ax.plot(epochs, coverage, color=COLORS['coverage'], linewidth=3,
                marker='o', markersize=6, markevery=3, label='Coverage')
        
        # Add target zone
        ax.axhspan(0.88, 0.905, alpha=0.2, color='green', label='Target Zone (88-90.5%)')
        ax.axhline(y=0.89, color=COLORS['target'], linestyle='--', linewidth=2, 
                  label='Target (89%)', alpha=0.8)
        
        # Statistics
        mean_cov = np.mean(coverage)
        std_cov = np.std(coverage)
        ax.text(0.98, 0.02, f'Mean: {mean_cov:.3f} ± {std_cov:.3f}', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Coverage Rate', fontweight='bold')
        ax.set_title('Coverage Evolution', fontweight='bold', pad=10)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.set_ylim(0.75, 1.0)
        ax.grid(True, alpha=0.3, linestyle='--')


def plot_mpiw_evolution(ax, history):
    """Plot MPIW evolution with trend."""
    mpiw = history.get('avg_mpiw', history.get('val_mpiw', []))
    
    if mpiw:
        epochs = range(1, len(mpiw) + 1)
        ax.plot(epochs, mpiw, color=COLORS['mpiw'], linewidth=3,
                marker='D', markersize=6, markevery=3, label='MPIW')
        
        # Statistics
        mean_mpiw = np.mean(mpiw)
        std_mpiw = np.std(mpiw)
        min_mpiw = np.min(mpiw)
        
        ax.text(0.98, 0.98, f'Mean: {mean_mpiw:.1f} ± {std_mpiw:.1f}\\nMin: {min_mpiw:.1f}', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('MPIW (pixels)', fontweight='bold')
        ax.set_title('Mean Prediction Interval Width', fontweight='bold', pad=10)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')


def plot_tau_evolution_subplot(ax, history):
    """Plot tau evolution in subplot."""
    tau = history.get('tau', [])
    
    if tau:
        epochs = range(1, len(tau) + 1)
        ax.plot(epochs, tau, color=COLORS['tau'], linewidth=3,
                marker='h', markersize=8, markevery=3, label='Tau')
        
        # Reference line
        ax.axhline(y=1.0, color='black', linestyle=':', linewidth=2, 
                  alpha=0.5, label='Reference (τ=1.0)')
        
        # Fill regions
        tau_array = np.array(tau)
        ax.fill_between(epochs, 1.0, tau_array, where=(tau_array > 1.0),
                       alpha=0.2, color='red', label='Over-correction')
        ax.fill_between(epochs, tau_array, 1.0, where=(tau_array < 1.0),
                       alpha=0.2, color='blue', label='Under-correction')
        
        # Final value
        ax.text(0.98, 0.02, f'Final τ: {tau[-1]:.4f}', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['tau'], alpha=0.3))
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Tau (τ)', fontweight='bold')
        ax.set_title('Calibration Factor Evolution', fontweight='bold', pad=10)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')


def plot_coverage_mpiw_tradeoff(ax, history):
    """Plot coverage vs MPIW tradeoff scatter."""
    coverage = history.get('coverage_rate', history.get('val_coverage', []))
    mpiw = history.get('avg_mpiw', history.get('val_mpiw', []))
    
    if coverage and mpiw and len(coverage) == len(mpiw):
        # Create scatter plot with epoch coloring
        scatter = ax.scatter(coverage, mpiw, c=range(len(coverage)), 
                           cmap='viridis', s=100, alpha=0.8, 
                           edgecolors='black', linewidth=1)
        
        # Mark start and end points
        ax.scatter(coverage[0], mpiw[0], color='green', s=300, 
                  marker='*', label='Start', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(coverage[-1], mpiw[-1], color='red', s=300, 
                  marker='*', label='End', edgecolors='black', linewidth=2, zorder=5)
        
        # Add ideal region
        rect = patches.Rectangle((0.88, 0), 0.025, max(mpiw) * 1.1, 
                               linewidth=0, facecolor='green', alpha=0.1)
        ax.add_patch(rect)
        ax.text(0.8925, max(mpiw) * 0.95, 'Ideal\\nRegion', ha='center', va='top',
                fontsize=10, alpha=0.7)
        
        # Target line
        ax.axvline(x=0.89, color=COLORS['target'], linestyle='--', 
                  linewidth=2, alpha=0.8)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Epoch', fontweight='bold')
        
        ax.set_xlabel('Coverage Rate', fontweight='bold')
        ax.set_ylabel('MPIW (pixels)', fontweight='bold')
        ax.set_title('Coverage-Efficiency Tradeoff', fontweight='bold', pad=10)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')


def plot_size_stratified_evolution(ax, history, metric='coverage'):
    """Plot size-stratified metrics over time."""
    size_metrics = history.get('size_metrics', [])
    
    if not size_metrics:
        return
    
    # Extract data
    epochs = range(1, len(size_metrics) + 1)
    small_data = []
    medium_data = []
    large_data = []
    
    for epoch_data in size_metrics:
        if isinstance(epoch_data, dict):
            small_data.append(epoch_data.get('small', {}).get(metric, 0))
            medium_data.append(epoch_data.get('medium', {}).get(metric, 0))
            large_data.append(epoch_data.get('large', {}).get(metric, 0))
    
    # Plot lines
    if small_data:
        ax.plot(epochs, small_data, color=COLORS['small'], linewidth=3,
                marker='o', markersize=6, markevery=3, label='Small (<32²)')
    if medium_data:
        ax.plot(epochs, medium_data, color=COLORS['medium'], linewidth=3,
                marker='s', markersize=6, markevery=3, label='Medium (32²-96²)')
    if large_data:
        ax.plot(epochs, large_data, color=COLORS['large'], linewidth=3,
                marker='^', markersize=6, markevery=3, label='Large (>96²)')
    
    # Add target lines for coverage
    if metric == 'coverage':
        ax.axhline(y=0.90, color=COLORS['small'], linestyle=':', linewidth=2, alpha=0.5)
        ax.axhline(y=0.89, color=COLORS['medium'], linestyle=':', linewidth=2, alpha=0.5)
        ax.axhline(y=0.85, color=COLORS['large'], linestyle=':', linewidth=2, alpha=0.5)
        ax.set_ylim(0.75, 1.0)
        ylabel = 'Coverage Rate'
        title = 'Size-Stratified Coverage Evolution'
    else:
        ylabel = 'MPIW (pixels)'
        title = 'Size-Stratified MPIW Evolution'
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(loc='right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')


def create_all_plots(
    history: Dict[str, List],
    output_dir: Path,
    model_name: str = "symmetric_adaptive"
):
    """Create all plots and save CSV."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Extract size metrics
    size_metrics = history.get('size_metrics', [])
    
    # Save CSV file
    csv_path = output_dir / f'{model_name}_training_history.csv'
    save_training_csv(history, size_metrics, csv_path, model_name)
    
    # Create comprehensive training plot
    plot_comprehensive_training_results(
        history,
        save_path=plots_dir / 'comprehensive_training_results.png',
        show=False,
        title_prefix=model_name.replace('_', ' ').title()
    )
    
    # Create standard training plot for backward compatibility
    plot_training_results(
        history,
        save_path=plots_dir / 'final_training_results.png',
        show=False
    )
    
    # Create final size comparison if we have the data
    if size_metrics and len(size_metrics) > 0:
        final_size_results = size_metrics[-1] if isinstance(size_metrics[-1], dict) else {}
        if final_size_results:
            plot_size_stratified_results(
                final_size_results,
                save_path=plots_dir / 'final_size_comparison.png',
                show=False
            )
    
    # Create tau evolution plot
    tau_history = history.get('tau', [])
    if tau_history:
        plot_tau_evolution(
            tau_history,
            coverage_history=history.get('coverage_rate', []),
            save_path=plots_dir / 'tau_evolution_detailed.png',
            show=False
        )
    
    print(f"\n✨ All plots saved to: {plots_dir}")
    print(f"📊 Training history CSV saved to: {csv_path}")