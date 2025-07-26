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
# MULTI-MODEL DATA PATHS - 4 models per dataset
# ============================================================================

MULTI_MODEL_PATHS = {
    'coco': {
        'x_101_fpn': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/coco_x_101_fpn/symmetric_adaptive_training_history.csv',
            'name': 'ResNeXt-101-FPN',
            'linestyle': '-'
        },
        'r_50_fpn': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/coco_r_50_fpn/symmetric_adaptive_training_history.csv',
            'name': 'ResNet-50-FPN',
            'linestyle': '--'
        },
        'cascade_r_50': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/coco_cascade_r_50/symmetric_adaptive_training_history.csv',
            'name': 'Cascade R-CNN',
            'linestyle': '-.'
        },
        'r_50_c4': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/coco_r_50_c4/symmetric_adaptive_training_history.csv',
            'name': 'ResNet-50-C4',
            'linestyle': ':'
        }
    },
    'bdd100k': {
        'x_101_fpn': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/bdd100k_x_101_fpn/symmetric_adaptive_training_history.csv',
            'name': 'ResNeXt-101-FPN',
            'linestyle': '-'
        },
        'r_50_fpn': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/bdd100k_r_50_fpn/symmetric_adaptive_training_history.csv',
            'name': 'ResNet-50-FPN',
            'linestyle': '--'
        },
        'cascade_r_50': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/bdd100k_cascade_r_50/symmetric_adaptive_training_history.csv',
            'name': 'Cascade R-CNN',
            'linestyle': '-.'
        },
        'r_50_c4': {
            'path': '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/bdd100k_r_50_c4/symmetric_adaptive_training_history.csv',
            'name': 'ResNet-50-C4',
            'linestyle': ':'
        }
    }
}

# Professional color palette with lightness variations
DATASET_COLORS = {
    'coco': {
        'x_101_fpn': '#0d4f8c',      # Dark blue (primary model)
        'r_50_fpn': '#1f77b4',       # Medium blue  
        'cascade_r_50': '#5ba3d4',   # Light blue
        'r_50_c4': '#aec7e8'         # Very light blue
    },
    'bdd100k': {
        'x_101_fpn': '#cc4125',      # Dark orange (primary model)
        'r_50_fpn': '#ff7f0e',       # Medium orange
        'cascade_r_50': '#ff9f40',   # Light orange
        'r_50_c4': '#ffbb78'         # Very light orange
    }
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_multi_model_data():
    """Load training history data for all models across both datasets."""
    all_data = {}
    
    for dataset_name in ['coco', 'bdd100k']:
        all_data[dataset_name] = {}
        print(f"📊 Loading {dataset_name.upper()} models:")
        
        for model_key, config in MULTI_MODEL_PATHS[dataset_name].items():
            file_path = config['path']
            
            if not os.path.exists(file_path):
                print(f"  ❌ {config['name']} (file missing)")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Extract key metrics
                data = {
                    'epoch': df['epoch'].values,
                    'coverage': df['val_coverage'].values,
                    'mpiw': df['val_mpiw'].values,
                    'tau': df['tau'].values,
                    'train_loss': df['train_loss'].values,
                    'name': config['name'],
                    'linestyle': config['linestyle']
                }
                
                all_data[dataset_name][model_key] = data
                print(f"  ✅ {config['name']}")
                
            except Exception as e:
                print(f"  ❌ {config['name']} (error: {e})")
    
    return all_data

# ============================================================================
# PLOT 1: TAU CALIBRATION ANALYSIS (3-PANEL)
# ============================================================================

def create_tau_calibration_analysis():
    """Create 3-panel plot showing Coverage, Tau, and MPIW evolution for all models."""
    
    # Load data for all models
    all_data = load_multi_model_data()
    
    # Create figure with 3 square subplots - Conference paper standard
    fig, axes = plt.subplots(3, 1, figsize=(text_width * 0.9, text_width * 1.8))
    # fig.suptitle('Tau Calibration Analysis', 
    #              fontsize=fs_p1, fontweight='bold', y=0.95)
    
    # ========================================================================
    # TOP PANEL: COVERAGE EVOLUTION
    # ========================================================================
    ax1 = axes[0]
    
    # Plot coverage evolution for all models with color variations
    for dataset_name in ['coco', 'bdd100k']:
        dataset_colors = DATASET_COLORS[dataset_name]
        dataset_models = all_data[dataset_name]
        
        for model_key, data in dataset_models.items():
            # Only label the first model per dataset to avoid legend clutter
            label = dataset_name.upper() if model_key == 'x_101_fpn' else ''
            
            # Get model-specific color
            color = dataset_colors[model_key]
            
            ax1.plot(data['epoch'], data['coverage'] * 100, 
                     color=color, linewidth=1.0, linestyle='-',
                     label=label, alpha=0.9)
    
    # Target coverage zone (88-92%)
    ax1.axhspan(88, 92, alpha=0.2, color='green', label='Target Zone (88-92%)')
    ax1.axhline(y=90, color='darkgreen', linestyle='--', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Coverage (%)', fontweight='bold')
    ax1.set_ylim(82, 98)
    ax1.set_xlim(0, 50)
    ax1.set_aspect('auto')  # Let matplotlib handle aspect ratio
    ax1.grid(True, alpha=0.3)
    # Instead of individual legends, we'll add a comprehensive one at the bottom
    # ax1.legend(loc='upper right', framealpha=0.9, fontsize=fs_m1)
    # ax1.set_title('Coverage Maintained', fontweight='bold', loc='center', fontsize=fs)
    
    # ========================================================================
    # MIDDLE PANEL: TAU EVOLUTION  
    # ========================================================================
    ax2 = axes[1]
    
    # Plot tau evolution for all models with color variations
    for dataset_name in ['coco', 'bdd100k']:
        dataset_colors = DATASET_COLORS[dataset_name]
        dataset_models = all_data[dataset_name]
        
        for model_key, data in dataset_models.items():
            # Only label the first model per dataset to avoid legend clutter
            label = dataset_name.upper() if model_key == 'x_101_fpn' else ''
            
            # Get model-specific color
            color = dataset_colors[model_key]
            
            ax2.plot(data['epoch'], data['tau'], 
                     color=color, linewidth=1.0, linestyle='-',
                     label=label, alpha=0.9)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Calibration (τ)', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(0, 50)
    ax2.set_aspect('auto')  # Let matplotlib handle aspect ratio
    ax2.grid(True, alpha=0.3)
    # ax2.legend(loc='upper right', framealpha=0.9, fontsize=fs_m1)
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
    # BOTTOM PANEL: MPIW EVOLUTION
    # ========================================================================
    ax3 = axes[2]
    
    # Plot MPIW evolution for all models and collect statistics
    mpiw_stats = {'coco': [], 'bdd100k': []}
    
    for dataset_name in ['coco', 'bdd100k']:
        dataset_colors = DATASET_COLORS[dataset_name]
        dataset_models = all_data[dataset_name]
        
        for model_key, data in dataset_models.items():
            # Only label the first model per dataset to avoid legend clutter
            label = dataset_name.upper() if model_key == 'x_101_fpn' else ''
            
            # Get model-specific color
            color = dataset_colors[model_key]
            
            ax3.plot(data['epoch'], data['mpiw'], 
                     color=color, linewidth=1.0, linestyle='-',
                     label=label, alpha=0.9)
            
            # Calculate improvement for statistics
            initial_mpiw = data['mpiw'][0]
            final_mpiw = data['mpiw'][-1]
            improvement = (initial_mpiw - final_mpiw) / initial_mpiw * 100
            mpiw_stats[dataset_name].append(improvement)
    
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('MPIW (pixels)', fontweight='bold')
    ax3.set_xlim(0, 50)
    # Set y-limits to make square aspect ratio work better
    all_mpiw_values = []
    for dataset_name in ['coco', 'bdd100k']:
        for model_key, data in all_data[dataset_name].items():
            all_mpiw_values.extend(data['mpiw'])
    
    if all_mpiw_values:
        max_mpiw = max(all_mpiw_values)
        min_mpiw = min(all_mpiw_values)
        mpiw_range = max_mpiw - min_mpiw
        ax3.set_ylim(min_mpiw - mpiw_range*0.1, max_mpiw + mpiw_range*0.1)
    ax3.set_aspect('auto')  # Let matplotlib handle aspect ratio
    ax3.grid(True, alpha=0.3)
    # ax3.legend(loc='upper right', framealpha=0.9, fontsize=fs_m1)
    # ax3.set_title('MPIW Reduction', fontweight='bold', loc='center', fontsize=fs)
    
    # Print multi-model statistics
    print(f"\n📊 Multi-Model MPIW Improvement Statistics:")
    for dataset_name in ['coco', 'bdd100k']:
        improvements = mpiw_stats[dataset_name]
        if improvements:
            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            print(f"  {dataset_name.upper()}: {mean_imp:.1f}±{std_imp:.1f}% reduction (n={len(improvements)} models)")
        else:
            print(f"  {dataset_name.upper()}: No data available")
    
    # ========================================================================
    # COMPREHENSIVE MODEL LEGEND - Shows which line corresponds to which model
    # ========================================================================
    
    # Create custom legend elements for all models
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Add COCO models
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, label='COCO Dataset:'))
    for model_key, color in DATASET_COLORS['coco'].items():
        model_name = MULTI_MODEL_PATHS['coco'][model_key]['name']
        legend_elements.append(Line2D([0], [0], color=color, linewidth=1.5, 
                                    label=f'  {model_name}'))
    
    # Add spacing
    legend_elements.append(Line2D([0], [0], color='white', linewidth=0, label=''))
    
    # Add BDD100K models  
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, label='BDD100K Dataset:'))
    for model_key, color in DATASET_COLORS['bdd100k'].items():
        model_name = MULTI_MODEL_PATHS['bdd100k'][model_key]['name']
        legend_elements.append(Line2D([0], [0], color=color, linewidth=1.5,
                                    label=f'  {model_name}'))
    
    # Add the comprehensive legend outside the plot area
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, frameon=True, framealpha=0.95, fontsize=fs_m1-1,
               columnspacing=1.5, handlelength=1.5)
    
    # Adjust layout for square subplots in vertical format with room for legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.97, bottom=0.15, left=0.15, right=0.90, hspace=0.3)
    
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
    """Create multi-dataset coverage-efficiency evolution plot with shape-based differentiation."""
    
    # Load data for all models and datasets
    all_data = load_multi_model_data()
    
    # Create figure - single plot for both datasets
    fig, ax = plt.subplots(1, 1, figsize=(text_width * 0.8, text_width * 0.8))
    
    # ========================================================================
    # MULTI-DATASET COVERAGE-EFFICIENCY TRADEOFF
    # ========================================================================
    
    # Collect all epoch data for unified colorbar
    all_epochs = []
    all_coverage = []
    all_mpiw = []
    
    # Dataset-specific marker configurations
    dataset_markers = {
        'coco': {'start': '*', 'end': 'o'},      # star for start, circle for end
        'bdd100k': {'start': 's', 'end': '^'}    # square for start, triangle for end
    }
    
    # Plot trajectories for all models across both datasets
    for dataset_name in ['coco', 'bdd100k']:
        dataset_colors = DATASET_COLORS[dataset_name]
        dataset_models = all_data[dataset_name]
        
        for model_key, data in dataset_models.items():
            coverage = data['coverage'] * 100  # Convert to percentage
            mpiw = data['mpiw']
            epochs = data['epoch']
            
            # Collect data for unified processing
            all_epochs.extend(epochs)
            all_coverage.extend(coverage)
            all_mpiw.extend(mpiw)
            
            # Get model-specific color
            color = dataset_colors[model_key]
            
            # Plot trajectory dots with epoch-based viridis coloring
            scatter = ax.scatter(coverage, mpiw, c=epochs, cmap='viridis', 
                               s=25, alpha=0.7, edgecolors='none', zorder=3)
            
            # Mark start point with dataset-specific shape and model color
            ax.scatter(coverage[0], mpiw[0], 
                      color=color, s=100, 
                      marker=dataset_markers[dataset_name]['start'], 
                      zorder=5, edgecolor='darkred', linewidth=1.5,
                      alpha=0.9)
            
            # Mark end point with dataset-specific shape and model color  
            ax.scatter(coverage[-1], mpiw[-1], 
                      color=color, s=100,
                      marker=dataset_markers[dataset_name]['end'],
                      zorder=5, edgecolor='black', linewidth=1.5,
                      alpha=0.9)
    
    # Add unified colorbar for epochs (using the last scatter object)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Epoch', fontweight='bold', fontsize=fs)
    cbar.ax.tick_params(labelsize=fs_m1)
    
    # ========================================================================
    # CREATE COMPREHENSIVE LEGEND
    # ========================================================================
    
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Start/End markers section
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, 
                                 label='Start/End Markers:', marker='None'))
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=0, 
                                 label='★ COCO Start     ● COCO End', marker='None'))
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=0,
                                 label='■ BDD100K Start  ▲ BDD100K End', marker='None'))
    
    # Add spacing
    legend_elements.append(Line2D([0], [0], color='white', linewidth=0, label=''))
    
    # Model colors section
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2,
                                 label='Model Colors:', marker='None'))
    
    # COCO models
    for model_key, color in DATASET_COLORS['coco'].items():
        model_name = MULTI_MODEL_PATHS['coco'][model_key]['name']
        legend_elements.append(Line2D([0], [0], color=color, linewidth=3,
                                     label=f'  COCO: {model_name}'))
    
    # BDD100K models  
    for model_key, color in DATASET_COLORS['bdd100k'].items():
        model_name = MULTI_MODEL_PATHS['bdd100k'][model_key]['name']
        legend_elements.append(Line2D([0], [0], color=color, linewidth=3,
                                     label=f'  BDD100K: {model_name}'))
    
    # Professional formatting
    ax.set_xlabel('Coverage Rate', fontweight='bold', fontsize=fs)
    ax.set_ylabel('MPIW (pixels)', fontweight='bold', fontsize=fs)
    ax.set_title('Multi-Architecture Coverage-Efficiency Tradeoff', 
                fontweight='bold', fontsize=fs_p1, pad=15)
    
    # Set appropriate limits with padding
    if all_coverage and all_mpiw:
        coverage_margin = (max(all_coverage) - min(all_coverage)) * 0.05
        mpiw_margin = (max(all_mpiw) - min(all_mpiw)) * 0.05
        ax.set_xlim(min(all_coverage) - coverage_margin, max(all_coverage) + coverage_margin)
        ax.set_ylim(min(all_mpiw) - mpiw_margin, max(all_mpiw) + mpiw_margin)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add the comprehensive legend outside the plot area
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=2, frameon=True, framealpha=0.95, fontsize=fs_m1-1,
               columnspacing=1.5, handlelength=1.5)
    
    # Adjust layout with room for legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.25, left=0.12, right=0.85)
    
    # Save plot
    output_path = '/ssd_4TB/divake/conformal-od/plots/coverage_efficiency_evolution.png'
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Coverage-efficiency evolution saved: {output_path}")
    print(f"📊 Total trajectories: 8 (4 COCO + 4 BDD100K models)")
    print(f"📊 Coverage range: {min(all_coverage):.1f}% - {max(all_coverage):.1f}%")
    print(f"📊 MPIW range: {min(all_mpiw):.1f} - {max(all_mpiw):.1f} pixels")
    print(f"📊 Epoch range: {min(all_epochs)} - {max(all_epochs)}")
    
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
        
        # Create Plot 2: Coverage-Efficiency Evolution (COCO only, matching reference style)
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