#!/usr/bin/env python3
import os
import yaml
import glob

# Define your output directory
OUTPUT_DIR = "/ssd_4TB/divake/conformal-od/output"
# Define your data directory
DATA_DIR = "/ssd_4TB/divake/conformal-od/data"

# Get all yaml files in config/coco_val
config_files = glob.glob("config/coco_val/*.yaml")

for config_file in config_files:
    print(f"Processing {config_file}")
    
    # Read the yaml file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the output directory
    config['PROJECT']['OUTPUT_DIR'] = OUTPUT_DIR
    
    # Update the dataset directory
    config['DATASETS']['DIR'] = DATA_DIR
    
    # Update MODEL.FILE path if it's a relative path
    if 'MODEL' in config and 'FILE' in config['MODEL']:
        if config['MODEL']['FILE'].startswith('COCO-Detection'):
            config['MODEL']['FILE'] = f"/ssd_4TB/divake/conformal-od/detectron2/configs/{config['MODEL']['FILE']}"
    
    # Check for ensemble model file paths
    if 'MODEL' in config and 'ENSEMBLE' in config['MODEL'] and 'FILE' in config['MODEL']['ENSEMBLE']:
        for i, file_path in enumerate(config['MODEL']['ENSEMBLE']['FILE']):
            if file_path.startswith('COCO-Detection'):
                config['MODEL']['ENSEMBLE']['FILE'][i] = f"/ssd_4TB/divake/conformal-od/detectron2/configs/{file_path}"
    
    # Write the updated yaml file
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

print("All config files updated!") 