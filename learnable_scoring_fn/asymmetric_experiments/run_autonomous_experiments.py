#!/usr/bin/env python
"""
Main launcher for autonomous asymmetric framework experiments.
This script manages the execution of all 5 frameworks sequentially.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append('/ssd_4TB/divake/conformal-od/learnable_scoring_fn')


def update_master_log(framework_name, status, metrics=None):
    """Update the master log with current progress."""
    master_log_path = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/unified_memory/master_log.json'
    
    with open(master_log_path, 'r') as f:
        master_log = json.load(f)
    
    master_log['current_framework'] = framework_name
    master_log['overall_progress']['total_experiments_run'] += 1
    
    if status == 'completed':
        master_log['frameworks_completed'].append(framework_name)
    elif status == 'failed':
        master_log['frameworks_failed'].append(framework_name)
    
    if metrics and metrics.get('mpiw_reduction', 0) > master_log['overall_progress']['best_mpiw_reduction']:
        master_log['overall_progress']['best_framework'] = framework_name
        master_log['overall_progress']['best_mpiw_reduction'] = metrics['mpiw_reduction']
    
    with open(master_log_path, 'w') as f:
        json.dump(master_log, f, indent=2)


def update_comparison_table(framework_name, results):
    """Update the comparison CSV table."""
    csv_path = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/unified_memory/comparison_table.csv'
    
    # Read existing table
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Create new row
    new_row = f"{framework_name},"
    new_row += f"{results.get('best_coverage', 0):.3f},"
    new_row += f"{results.get('best_mpiw_small', 0):.1f},"
    new_row += f"{results.get('best_mpiw_medium', 0):.1f},"
    new_row += f"{results.get('best_mpiw_large', 0):.1f},"
    new_row += f"{results.get('best_mpiw', 0):.1f},"
    new_row += f"{results.get('mpiw_reduction', 0):.1f}%,"
    new_row += f"{results.get('status', 'unknown')},"
    new_row += f"{results.get('total_attempts', 0)},"
    new_row += f"{format_time(results.get('total_time', 0))}\n"
    
    # Append new row
    lines.append(new_row)
    
    # Write back
    with open(csv_path, 'w') as f:
        f.writelines(lines)


def format_time(seconds):
    """Format seconds into hours and minutes."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes}m"


def commit_to_github(framework_name, results):
    """Commit framework results to GitHub."""
    os.chdir('/ssd_4TB/divake/conformal-od')
    
    # Stage changes
    subprocess.run(['git', 'add', 'learnable_scoring_fn/asymmetric_experiments/'], check=True)
    
    # Create commit message
    commit_message = f"F_03_GAP: {framework_name} framework - "
    if results.get('status') == 'completed':
        commit_message += f"Success! {results.get('best_coverage', 0):.1%} coverage, "
        commit_message += f"{results.get('mpiw_reduction', 0):.1f}% MPIW reduction"
    else:
        commit_message += f"Failed after {results.get('total_attempts', 0)} attempts"
    
    commit_message += f"\n\n"
    commit_message += f"- Best coverage: {results.get('best_coverage', 0):.3f}\n"
    commit_message += f"- Best MPIW: {results.get('best_mpiw', 0):.1f}\n"
    commit_message += f"- MPIW reduction: {results.get('mpiw_reduction', 0):.1f}%\n"
    commit_message += f"- Total attempts: {results.get('total_attempts', 0)}\n"
    commit_message += f"- Time spent: {format_time(results.get('total_time', 0))}"
    
    # Commit
    subprocess.run(['git', 'commit', '--no-gpg-sign', '-m', commit_message], check=True)
    
    # Push
    subprocess.run(['git', 'push'], check=True)
    
    print(f"Committed and pushed results for {framework_name}")


def run_framework(framework_name, framework_number):
    """Run a specific framework."""
    print(f"\n{'='*80}")
    print(f"Starting Framework {framework_number}: {framework_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Update master log
    update_master_log(framework_name, 'in_progress')
    
    # Run the framework
    if framework_name == 'GAP':
        # Run GAP training
        os.chdir('/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/frameworks/GAP')
        cmd = [
            '/home/divake/miniconda3/envs/env_cu121/bin/python',
            'train_gap.py'
        ]
        
        try:
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load results from framework memory
                memory_path = f'/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/unified_memory/framework_memories/framework_{framework_number}_{framework_name}_memory.json'
                with open(memory_path, 'r') as f:
                    framework_memory = json.load(f)
                
                if framework_memory.get('best_result'):
                    results = framework_memory['best_result']
                    results['status'] = 'completed'
                    results['total_attempts'] = framework_memory['total_attempts']
                    results['total_time'] = sum(attempt.get('total_time', 0) for attempt in framework_memory['attempts'])
                    
                    # Update logs and tables
                    update_master_log(framework_name, 'completed', results)
                    update_comparison_table(framework_name, results)
                    
                    # Commit to GitHub
                    commit_to_github(framework_name, results)
                    
                    return True
                else:
                    print(f"Framework {framework_name} failed to achieve target")
                    results = {
                        'status': 'failed',
                        'total_attempts': framework_memory.get('total_attempts', 0),
                        'best_coverage': 0,
                        'best_mpiw': float('inf'),
                        'mpiw_reduction': 0
                    }
                    update_master_log(framework_name, 'failed')
                    update_comparison_table(framework_name, results)
                    commit_to_github(framework_name, results)
                    return False
            else:
                print(f"Error running {framework_name}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Exception running {framework_name}: {e}")
            return False
    
    else:
        # Placeholder for other frameworks
        print(f"Framework {framework_name} not yet implemented")
        return False


def main():
    """Main execution function."""
    print("="*80)
    print("AUTONOMOUS ASYMMETRIC FRAMEWORK EXPERIMENTS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: 88-92% coverage with maximum MPIW reduction")
    print("="*80)
    
    # Framework list
    frameworks = [
        ('GAP', 1),
        ('OAGA', 2),
        ('MAP', 3),
        ('MoE_EW', 4),
        ('HMAP', 5)
    ]
    
    # Run each framework
    for framework_name, framework_number in frameworks:
        success = run_framework(framework_name, framework_number)
        
        if not success:
            print(f"\nWarning: Framework {framework_name} did not achieve target")
        
        # Small delay between frameworks
        time.sleep(10)
    
    # Final summary
    print("\n" + "="*80)
    print("ALL FRAMEWORKS COMPLETED")
    print("="*80)
    
    # Load and display final results
    master_log_path = '/ssd_4TB/divake/conformal-od/learnable_scoring_fn/asymmetric_experiments/unified_memory/master_log.json'
    with open(master_log_path, 'r') as f:
        master_log = json.load(f)
    
    print(f"Best framework: {master_log['overall_progress']['best_framework']}")
    print(f"Best MPIW reduction: {master_log['overall_progress']['best_mpiw_reduction']:.1f}%")
    print(f"Frameworks completed: {', '.join(master_log['frameworks_completed'])}")
    print(f"Frameworks failed: {', '.join(master_log['frameworks_failed'])}")
    print("="*80)


if __name__ == "__main__":
    main()