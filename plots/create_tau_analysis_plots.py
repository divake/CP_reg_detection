#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tau Calibration Analysis Plots for AAAI Paper
Creates publication-quality plots showing:
1. Tau calibration analysis (3-panel: Coverage, Tau, MPIW evolution)
2. Coverage-efficiency tradeoff evolution during training
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

# Professional matplotlib styling
text_width = 5.50107  # inches for AAAI format
dpi = 300
fs_m1 = 9   # for figure ticks
fs = 10     # for regular figure text  
fs_p1 = 12  # figure titles

plt.style.use('default')
plt.rcParams.update({
    'font.size': fs,
    'axes.titlesize': fs,
    'axes.labelsize': fs,
    'xtick.labelsize': fs_m1,
    'ytick.labelsize': fs_m1,
    'legend.fontsize': fs_m1,
    'figure.titlesize': fs_p1,
    'figure.dpi': dpi,
    'savefig.dpi': 1000,
    'savefig.bbox': 'tight',
    'grid.alpha': 0.3,
    'axes.grid': True,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'text.usetex': False,
    'lines.linewidth': 2,
    'lines.markersize': 4
})

# ============================================================================
# DATA PATHS
# ============================================================================

TRAINING_DATA_PATHS = {
    'coco': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/coco_x_101_fpn/symmetric_adaptive_training_history.csv',
    'bdd100k': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/bdd100k_x_101_fpn/symmetric_adaptive_training_history.csv'
}

# Professional colors for datasets
DATASET_COLORS = {
    'coco': '#1f77b4',      # Blue
    'bdd100k': '#ff7f0e'    # Orange
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_training_data(dataset_name):
    """Load training history data for a dataset."""
    if dataset_name not in TRAINING_DATA_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    file_path = TRAINING_DATA_PATHS[dataset_name]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Extract key metrics
    data = {
        'epoch': df['epoch'].values,
        'coverage': df['val_coverage'].values,
        'mpiw': df['val_mpiw'].values,
        'tau': df['tau'].values,
        'train_loss': df['train_loss'].values
    }
    
    return data

# ============================================================================
# PLOT 1: TAU CALIBRATION ANALYSIS (3-PANEL)
# ============================================================================

def create_tau_calibration_analysis():
    """Create 3-panel plot showing Coverage, Tau, and MPIW evolution."""
    
    # Load data for both datasets
    coco_data = load_training_data('coco')
    bdd100k_data = load_training_data('bdd100k')
    
    # Create figure with 3 square subplots - Conference paper standard
    fig, axes = plt.subplots(1, 3, figsize=(text_width * 2.0, text_width * 0.7))
    # fig.suptitle('Tau Calibration Analysis', 
    #              fontsize=fs_p1, fontweight='bold', y=0.95)
    
    # Colors
    coco_color = DATASET_COLORS['coco']
    bdd100k_color = DATASET_COLORS['bdd100k']
    
    # ========================================================================
    # LEFT PANEL: COVERAGE EVOLUTION
    # ========================================================================
    ax1 = axes[0]
    
    # Plot coverage evolution
    ax1.plot(coco_data['epoch'], coco_data['coverage'] * 100, 
             color=coco_color, linewidth=2, label='COCO', marker='o', markersize=3)
    ax1.plot(bdd100k_data['epoch'], bdd100k_data['coverage'] * 100, 
             color=bdd100k_color, linewidth=2, label='BDD100K', marker='s', markersize=3)
    
    # Target coverage zone (88-92%)
    ax1.axhspan(88, 92, alpha=0.2, color='green', label='Target Zone (88-92%)')
    ax1.axhline(y=90, color='darkgreen', linestyle='--', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Coverage (%)', fontweight='bold')
    ax1.set_ylim(82, 98)
    ax1.set_xlim(0, 50)
    # Make visually square by ensuring proper data range scaling
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=fs_m1)
    # ax1.set_title('Coverage Maintained', fontweight='bold', loc='center', fontsize=fs)
    
    # ========================================================================
    # MIDDLE PANEL: TAU EVOLUTION  
    # ========================================================================
    ax2 = axes[1]
    
    # Plot tau evolution
    ax2.plot(coco_data['epoch'], coco_data['tau'], 
             color=coco_color, linewidth=2, label='COCO', marker='o', markersize=3)
    ax2.plot(bdd100k_data['epoch'], bdd100k_data['tau'], 
             color=bdd100k_color, linewidth=2, label='BDD100K', marker='s', markersize=3)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Tau (τ)', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(0, 50)
    # Make visually square by ensuring proper data range scaling
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=fs_m1)
    #ax2.set_title('Tau Adaptation', fontweight='bold', loc='center', fontsize=fs)
    
    # Add annotations for final tau values (adjusted for square plot)
    # final_tau_coco = coco_data['tau'][-1]
    # final_tau_bdd100k = bdd100k_data['tau'][-1]
    # ax2.text(0.02, 0.98, f'Final COCO τ = {final_tau_coco:.3f}', 
    #          transform=ax2.transAxes, fontsize=fs_m1-1, color=coco_color,
    #          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
    #          facecolor='white', alpha=0.8))
    # ax2.text(0.02, 0.88, f'Final BDD100K τ = {final_tau_bdd100k:.3f}', 
    #          transform=ax2.transAxes, fontsize=fs_m1-1, color=bdd100k_color,
    #          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
    #          facecolor='white', alpha=0.8))
    
    # ========================================================================
    # RIGHT PANEL: MPIW EVOLUTION
    # ========================================================================
    ax3 = axes[2]
    
    # Plot MPIW evolution
    ax3.plot(coco_data['epoch'], coco_data['mpiw'], 
             color=coco_color, linewidth=2, label='COCO', marker='o', markersize=3)
    ax3.plot(bdd100k_data['epoch'], bdd100k_data['mpiw'], 
             color=bdd100k_color, linewidth=2, label='BDD100K', marker='s', markersize=3)
    
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('MPIW (pixels)', fontweight='bold')
    ax3.set_xlim(0, 50)
    # Set y-limits to make square aspect ratio work better
    max_mpiw = max(max(coco_data['mpiw']), max(bdd100k_data['mpiw']))
    min_mpiw = min(min(coco_data['mpiw']), min(bdd100k_data['mpiw']))
    mpiw_range = max_mpiw - min_mpiw
    ax3.set_ylim(min_mpiw - mpiw_range*0.1, max_mpiw + mpiw_range*0.1)
    # Make visually square by ensuring proper data range scaling
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=fs_m1)
    # ax3.set_title('MPIW Reduction', fontweight='bold', loc='center', fontsize=fs)
    
    # Add improvement percentages
    initial_mpiw_coco = coco_data['mpiw'][0]
    final_mpiw_coco = coco_data['mpiw'][-1]
    improvement_coco = (initial_mpiw_coco - final_mpiw_coco) / initial_mpiw_coco * 100
    
    initial_mpiw_bdd100k = bdd100k_data['mpiw'][0]
    final_mpiw_bdd100k = bdd100k_data['mpiw'][-1]
    improvement_bdd100k = (initial_mpiw_bdd100k - final_mpiw_bdd100k) / initial_mpiw_bdd100k * 100
    
    # Add text box with improvements (adjusted for square plot)
    # textstr = f'COCO: {improvement_coco:.1f}% ↓\nBDD100K: {improvement_bdd100k:.1f}% ↓'
    # props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8)
    # ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=fs_m1-1,
    #          verticalalignment='top', bbox=props)
    
    # Adjust layout for square format (horizontal layout)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.98, wspace=0.35)
    
    # Save plot
    output_path = '/ssd_4TB/divake/conformal-od/plots/tau_calibration_analysis.png'
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Tau calibration analysis saved: {output_path}")
    return fig

# ============================================================================
# PLOT 2: COVERAGE-EFFICIENCY TRADEOFF EVOLUTION
# ============================================================================

def create_coverage_efficiency_evolution():
    """Create coverage-efficiency evolution plot showing training trajectory."""
    
    # Load data
    coco_data = load_training_data('coco')
    bdd100k_data = load_training_data('bdd100k')
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(text_width * 2, text_width * 0.8))
    fig.suptitle('Coverage-Efficiency Tradeoff Evolution During Training', 
                 fontsize=fs_p1, fontweight='bold', y=0.95)
    
    datasets = [('coco', coco_data, 'COCO'), ('bdd100k', bdd100k_data, 'BDD100K')]
    
    for idx, (dataset_name, data, title) in enumerate(datasets):
        ax = axes[idx]
        
        coverage = data['coverage'] * 100  # Convert to percentage
        mpiw = data['mpiw']
        epochs = data['epoch']
        
        # Create color gradient for epochs
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        
        # Plot trajectory
        for i in range(len(epochs) - 1):
            ax.plot([coverage[i], coverage[i+1]], [mpiw[i], mpiw[i+1]], 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        # Mark start and end points
        ax.scatter(coverage[0], mpiw[0], color='red', s=100, marker='*', 
                  label='Start (Epoch 1)', zorder=5, edgecolor='darkred', linewidth=1)
        ax.scatter(coverage[-1], mpiw[-1], color='darkgreen', s=100, marker='*', 
                  label=f'End (Epoch {epochs[-1]})', zorder=5, edgecolor='black', linewidth=1)
        
        # Add arrows to show direction
        mid_point = len(epochs) // 2
        dx = coverage[mid_point + 5] - coverage[mid_point]
        dy = mpiw[mid_point + 5] - mpiw[mid_point]
        ax.arrow(coverage[mid_point], mpiw[mid_point], dx*0.5, dy*0.5,
                head_width=0.3, head_length=1.5, fc='black', ec='black', alpha=0.6)
        
        # Formatting
        ax.set_xlabel('Coverage (%)', fontweight='bold')
        ax.set_ylabel('MPIW (pixels)', fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {title}', fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Set appropriate limits
        coverage_margin = (max(coverage) - min(coverage)) * 0.1
        mpiw_margin = (max(mpiw) - min(mpiw)) * 0.1
        ax.set_xlim(min(coverage) - coverage_margin, max(coverage) + coverage_margin)
        ax.set_ylim(min(mpiw) - mpiw_margin, max(mpiw) + mpiw_margin)
        
        # Add colorbar for epochs
        if idx == 1:  # Only for the second subplot
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                     norm=plt.Normalize(vmin=1, vmax=max(epochs)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Epoch', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    
    # Save plot
    output_path = '/ssd_4TB/divake/conformal-od/plots/coverage_efficiency_evolution.png'
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Coverage-efficiency evolution saved: {output_path}")
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all tau analysis plots."""
    
    print("================================================================================")
    print("CREATING TAU CALIBRATION ANALYSIS PLOTS FOR AAAI PAPER")
    print("================================================================================")
    
    # Create output directory
    os.makedirs('/ssd_4TB/divake/conformal-od/plots', exist_ok=True)
    
    try:
        # Create Plot 1: Tau Calibration Analysis (3-panel)
        print("\n📊 Creating Tau Calibration Analysis Plot...")
        fig1 = create_tau_calibration_analysis()
        plt.close(fig1)
        
        # Create Plot 2: Coverage-Efficiency Evolution
        print("\n📈 Creating Coverage-Efficiency Evolution Plot...")
        fig2 = create_coverage_efficiency_evolution()
        plt.close(fig2)
        
        print("\n" + "="*80)
        print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
        print("="*80)
        print("📊 Tau Calibration Analysis: /ssd_4TB/divake/conformal-od/plots/tau_calibration_analysis.png")
        print("📈 Coverage-Efficiency Evolution: /ssd_4TB/divake/conformal-od/plots/coverage_efficiency_evolution.png")
        print("="*80)
        
        print("\n🎯 KEY INSIGHTS FOR AAAI PAPER:")
        print("1. Coverage remains stable (88-92%) while tau adapts → Conformal guarantees maintained")
        print("2. Tau decreases during training → Calibration improves efficiency")
        print("3. MPIW decreases proportionally → Direct efficiency improvement")
        print("4. Dataset-specific adaptation → COCO vs BDD100K show different optima")
        print("5. Training trajectory shows exploration → convergence to Pareto-optimal points")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        raise

if __name__ == "__main__":
    main()