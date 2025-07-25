#!/usr/bin/env python
"""
Parse Cityscapes polygon annotations to create bounding box ground truth.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def parse_cityscapes_to_coco_format():
    """Convert Cityscapes polygon annotations to COCO-like format with bboxes."""
    
    # Cityscapes categories we care about for object detection
    target_categories = {
        'person': 1,
        'rider': 2, 
        'car': 3,
        'truck': 4,
        'bus': 5,
        'train': 6,
        'motorcycle': 7,
        'bicycle': 8
    }
    
    # Setup paths
    val_gtfine_dir = "/ssd_4TB/divake/conformal-od/data/cityscapes/gtFine/val"
    val_img_dir = "/ssd_4TB/divake/conformal-od/data/cityscapes/leftImg8bit/val"
    
    # Collect all annotation files
    ann_files = []
    for city_dir in Path(val_gtfine_dir).glob("*"):
        if city_dir.is_dir():
            ann_files.extend(list(city_dir.glob("*_polygons.json")))
    
    print(f"Found {len(ann_files)} annotation files")
    
    # Build COCO-like structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in target_categories.items()]
    }
    
    img_id = 1
    ann_id = 1
    
    for ann_file in tqdm(ann_files, desc="Processing annotations"):
        # Read polygon data
        with open(ann_file, 'r') as f:
            poly_data = json.load(f)
        
        # Get image info
        img_name = ann_file.stem.replace("_gtFine_polygons", "_leftImg8bit.png")
        city = ann_file.parent.name
        img_path = Path(val_img_dir) / city / img_name
        
        if not img_path.exists():
            continue
            
        # Add image entry
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_name,
            "city": city,
            "height": poly_data["imgHeight"],
            "width": poly_data["imgWidth"]
        })
        
        # Process objects
        for obj in poly_data["objects"]:
            label = obj["label"]
            
            # Skip if not a target category
            if label not in target_categories:
                continue
                
            # Convert polygon to bbox
            polygon = obj["polygon"]
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Skip very small objects
            if width < 5 or height < 5:
                continue
            
            # Add annotation
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": target_categories[label],
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1
        
        img_id += 1
    
    # Save the converted annotations
    output_file = "/ssd_4TB/divake/conformal-od/data/cityscapes/cityscapes_val_coco_format.json"
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parse_cityscapes_to_coco_format()