#!/usr/bin/env python3
"""
Generate cache model from Faster R-CNN checkpoint.

This script:
1. Loads the Faster R-CNN model from checkpoint
2. Runs inference on COCO dataset (train and validation)
3. Extracts features using FeatureExtractor
4. Saves cache in the expected format for learnable scoring function training

Usage:
    python generate_cache.py --checkpoint /path/to/faster_rcnn_X_101_32x8d_FPN_3x.pth
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory and detectron2 to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "detectron2"))

# Import detectron2 components
try:
    # Import detectron2 using the same pattern as your working code
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.structures import Boxes, Instances
    from detectron2.utils.logger import setup_logger
    from detectron2.data import build_detection_train_loader, build_detection_test_loader
    from detectron2.data.dataset_mapper import DatasetMapper
    from detectron2.data.transforms import ResizeShortestEdge
    from detectron2.data.build import get_detection_dataset_dicts
    from detectron2.utils.visualizer import Visualizer
    print("Detectron2 imported successfully")
except ImportError as e:
    print(f"Error importing detectron2: {e}")
    print("Please check that all dependencies are properly installed in your environment.")
    sys.exit(1)

# Import feature extractor from learnable_scoring_fn
try:
    from learnable_scoring_fn.feature_utils import FeatureExtractor
    print("Feature extractor imported successfully")
except ImportError as e:
    print(f"Error importing feature extractor: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

# Setup logger
setup_logger()


class CacheGenerator:
    """Generate cache from Faster R-CNN checkpoint for learnable scoring function."""
    
    def __init__(self, checkpoint_path: str, coco_data_dir: str, output_dir: str, 
                 device: str = "auto", confidence_threshold: float = 0.05):
        """
        Initialize cache generator.
        
        Args:
            checkpoint_path: Path to Faster R-CNN checkpoint
            coco_data_dir: Path to COCO dataset directory
            output_dir: Output directory for cache files
            device: Device to use ('auto', 'cpu', 'cuda')
            confidence_threshold: Minimum confidence threshold for predictions
        """
        self.checkpoint_path = checkpoint_path
        self.coco_data_dir = Path(coco_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.confidence_threshold = confidence_threshold
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(img_height=480, img_width=640)
        
        # Initialize model
        self.model = None
        self.predictor = None
        
        print(f"Using device: {self.device}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"COCO data directory: {coco_data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def setup_model(self):
        """Setup the Faster R-CNN model from checkpoint."""
        print("Setting up Faster R-CNN model...")
        
        # Create config
        cfg = get_cfg()
        
        # Use the config for Faster R-CNN X-101-32x8d-FPN
        # Get the project root directory (parent of scripts/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, "detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.merge_from_file(config_path)
        
        # Set the checkpoint path
        cfg.MODEL.WEIGHTS = self.checkpoint_path
        
        # Set confidence threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        
        # Set device
        cfg.MODEL.DEVICE = self.device
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.model = self.predictor.model
        
        print("Model setup completed")
        print(f"Model device: {next(self.model.parameters()).device}")
    
    def register_coco_datasets(self):
        """Register COCO datasets with detectron2."""
        print("Registering COCO datasets...")
        
        # Register train dataset
        train_json = self.coco_data_dir / "annotations" / "instances_train2017.json"
        train_images = self.coco_data_dir / "train2017"
        
        if train_json.exists() and train_images.exists():
            register_coco_instances("coco_train", {}, str(train_json), str(train_images))
            print(f"Registered train dataset: {len(DatasetCatalog.get('coco_train'))} images")
        else:
            print(f"Warning: Train dataset not found at {train_json} or {train_images}")
        
        # Register val dataset
        val_json = self.coco_data_dir / "annotations" / "instances_val2017.json"
        val_images = self.coco_data_dir / "val2017"
        
        if val_json.exists() and val_images.exists():
            register_coco_instances("coco_val", {}, str(val_json), str(val_images))
            print(f"Registered val dataset: {len(DatasetCatalog.get('coco_val'))} images")
        else:
            print(f"Warning: Val dataset not found at {val_json} or {val_images}")
    
    def run_inference_on_dataset(self, dataset_name: str, max_images: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Run inference on a dataset and extract predictions and labels.
        
        Args:
            dataset_name: Name of registered dataset
            max_images: Maximum number of images to process (for testing)
            
        Returns:
            predictions: List of prediction dictionaries
            labels: List of label dictionaries
        """
        print(f"Running inference on {dataset_name}...")
        
        # Get dataset
        dataset_dicts = get_detection_dataset_dicts([dataset_name])
        
        if max_images:
            dataset_dicts = dataset_dicts[:max_images]
        
        predictions = []
        labels = []
        
        # Process each image
        for idx, record in enumerate(tqdm(dataset_dicts, desc=f"Processing {dataset_name}")):
            try:
                # Load image
                image_path = record["file_name"]
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, skipping...")
                    continue
                
                # Read image
                import cv2
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}, skipping...")
                    continue
                
                # Run inference
                outputs = self.predictor(image)
                
                # Extract predictions
                instances = outputs["instances"]
                if len(instances) == 0:
                    continue
                
                # Get predictions on CPU
                pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                pred_scores = instances.scores.cpu().numpy()
                
                # Create prediction dictionary
                pred_dict = {
                    'pred_coords': pred_boxes,  # [N, 4] - x0, y0, x1, y1
                    'pred_cls': pred_classes,   # [N] - class indices
                    'pred_score': pred_scores,  # [N] - confidence scores
                    'img_id': record.get('image_id', idx),
                    'height': record.get('height', image.shape[0]),
                    'width': record.get('width', image.shape[1])
                }
                
                # Extract ground truth
                gt_boxes = []
                gt_classes = []
                
                for annotation in record.get('annotations', []):
                    # Convert COCO bbox format [x, y, width, height] to [x0, y0, x1, y1]
                    bbox = annotation['bbox']
                    x0, y0, w, h = bbox
                    x1, y1 = x0 + w, y0 + h
                    
                    gt_boxes.append([x0, y0, x1, y1])
                    gt_classes.append(annotation['category_id'])
                
                # Create label dictionary
                label_dict = {
                    'gt_coords': np.array(gt_boxes) if gt_boxes else np.empty((0, 4)),
                    'gt_cls': np.array(gt_classes) if gt_classes else np.empty((0,)),
                    'img_id': record.get('image_id', idx),
                    'height': record.get('height', image.shape[0]),
                    'width': record.get('width', image.shape[1])
                }
                
                predictions.append(pred_dict)
                labels.append(label_dict)
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        print(f"Processed {len(predictions)} images from {dataset_name}")
        return predictions, labels
    
    def match_predictions_to_ground_truth(self, predictions: List[Dict], labels: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Match predictions to ground truth using IoU and create matched pairs.
        
        Args:
            predictions: List of prediction dictionaries
            labels: List of label dictionaries
            
        Returns:
            matched_predictions: List of matched prediction instances
            matched_labels: List of matched label instances
        """
        print("Matching predictions to ground truth...")
        
        def compute_iou(box1, box2):
            """Compute IoU between two boxes."""
            x1, y1, x2, y2 = box1
            x1_gt, y1_gt, x2_gt, y2_gt = box2
            
            # Intersection
            xi1, yi1 = max(x1, x1_gt), max(y1, y1_gt)
            xi2, yi2 = min(x2, x2_gt), min(y2, y2_gt)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0
            
            inter_area = (xi2 - xi1) * (yi2 - yi1)
            
            # Union
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        matched_predictions = []
        matched_labels = []
        
        iou_threshold = 0.5
        
        for pred_dict, label_dict in zip(predictions, labels):
            if pred_dict['img_id'] != label_dict['img_id']:
                continue
            
            pred_boxes = pred_dict['pred_coords']
            gt_boxes = label_dict['gt_coords']
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # For each prediction, find best matching ground truth
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                # If IoU is above threshold, create matched pair
                if best_iou > iou_threshold:
                    matched_pred = {
                        'pred_coords': pred_box,
                        'pred_cls': pred_dict['pred_cls'][i],
                        'pred_score': pred_dict['pred_score'][i],
                        'img_id': pred_dict['img_id'],
                        'height': pred_dict['height'],
                        'width': pred_dict['width']
                    }
                    
                    matched_label = {
                        'gt_coords': gt_boxes[best_gt_idx],
                        'gt_cls': label_dict['gt_cls'][best_gt_idx],
                        'img_id': label_dict['img_id'],
                        'height': label_dict['height'],
                        'width': label_dict['width'],
                        'iou': best_iou
                    }
                    
                    matched_predictions.append(matched_pred)
                    matched_labels.append(matched_label)
        
        print(f"Created {len(matched_predictions)} matched prediction-label pairs")
        return matched_predictions, matched_labels
    
    def extract_features(self, predictions: List[Dict]) -> torch.Tensor:
        """
        Extract 17-dimensional features from predictions (compatible with trained models).
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            features: [N, 17] tensor of extracted features (13 geometric + 4 uncertainty)
        """
        print("Extracting features from predictions...")
        
        if not predictions:
            return torch.empty(0, 17)
        
        n_predictions = len(predictions)
        features = torch.zeros(n_predictions, 17, dtype=torch.float32)
        
        for i, pred in enumerate(predictions):
            box = pred['pred_coords']
            score = pred['pred_score']
            img_h = pred['height']
            img_w = pred['width']
            
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Extract 13 features matching feature_utils.py
            # 1-4: Raw coordinates
            features[i, 0] = float(x1)                                     # x0
            features[i, 1] = float(y1)                                     # y0
            features[i, 2] = float(x2)                                     # x1
            features[i, 3] = float(y2)                                     # y1
            
            # 5: Confidence score
            features[i, 4] = float(score)                                  # confidence
            
            # 6: Log area
            area = float(w * h)
            features[i, 5] = float(np.log(max(area, 1e-6)))               # log_area
            
            # 7: Aspect ratio
            features[i, 6] = float(w / (h + 1e-6))                         # aspect_ratio
            
            # 8-9: Normalized center coordinates
            features[i, 7] = float((x1 + x2) / 2 / img_w)                 # center_x_norm
            features[i, 8] = float((y1 + y2) / 2 / img_h)                 # center_y_norm
            
            # 10-11: Position relative to image center
            features[i, 9] = float(((x1 + x2) / 2 - img_w / 2) / img_w)   # rel_pos_x
            features[i, 10] = float(((y1 + y2) / 2 - img_h / 2) / img_h)  # rel_pos_y
            
            # 12: Relative size
            features[i, 11] = float(area / (img_w * img_h))                # rel_size
            
            # 13: Distance to nearest edge (minimum of all 4 edges)
            dist_left = float(x1 / img_w)
            dist_right = float((img_w - x2) / img_w)
            dist_top = float(y1 / img_h)
            dist_bottom = float((img_h - y2) / img_h)
            features[i, 12] = float(min(dist_left, dist_right, dist_top, dist_bottom))  # edge_distance
            
            # 14-17: Uncertainty features (matching UncertaintyFeatureExtractor)
            # 14: Confidence-based uncertainty
            features[i, 13] = 1.0 - float(score)                           # uncertainty_score
            
            # 15: Ensemble uncertainty proxy (scaled confidence-based)
            features[i, 14] = (1.0 - float(score)) * 10.0                  # scaled uncertainty
            
            # 16: Expected error proxy (scaled by typical error magnitude)
            features[i, 15] = (1.0 - float(score)) * 50.0                  # expected_error
            
            # 17: Difficulty score (area difficulty + aspect ratio difficulty)
            area_difficulty = float(1.0 / (area + 1.0))
            aspect_difficulty = float(abs(np.log(float(w) / (float(h) + 1e-6) + 1e-6)))
            features[i, 16] = float((area_difficulty + aspect_difficulty) / 2.0)  # difficulty_score
        
        print(f"Extracted features shape: {features.shape}")
        return features
    
    def save_cache(self, train_predictions: List[Dict], train_labels: List[Dict],
                   val_predictions: List[Dict], val_labels: List[Dict],
                   train_features: torch.Tensor, val_features: torch.Tensor):
        """
        Save cache in the expected format.
        
        Args:
            train_predictions: Training predictions
            train_labels: Training labels
            val_predictions: Validation predictions
            val_labels: Validation labels
            train_features: Training features
            val_features: Validation features
        """
        print("Saving cache files...")
        
        # Save predictions as pickle files
        with open(self.output_dir / "predictions_train.pkl", 'wb') as f:
            pickle.dump((train_predictions, train_labels), f)
        
        with open(self.output_dir / "predictions_val.pkl", 'wb') as f:
            pickle.dump((val_predictions, val_labels), f)
        
        # Prepare data for .pt files
        train_data = {
            'features': train_features,
            'pred_coords': torch.tensor([p['pred_coords'] for p in train_predictions], dtype=torch.float32),
            'gt_coords': torch.tensor([l['gt_coords'] for l in train_labels], dtype=torch.float32),
            'confidence': torch.tensor([p['pred_score'] for p in train_predictions], dtype=torch.float32)
        }
        
        val_data = {
            'features': val_features,
            'pred_coords': torch.tensor([p['pred_coords'] for p in val_predictions], dtype=torch.float32),
            'gt_coords': torch.tensor([l['gt_coords'] for l in val_labels], dtype=torch.float32),
            'confidence': torch.tensor([p['pred_score'] for p in val_predictions], dtype=torch.float32)
        }
        
        # Save .pt files
        torch.save(train_data, self.output_dir / "features_train.pt")
        torch.save(val_data, self.output_dir / "features_val.pt")
        
        # Save features without image IDs (if needed)
        torch.save(train_features, self.output_dir / "features_train_no_img_ids.pt")
        torch.save(val_features, self.output_dir / "features_val_no_img_ids.pt")
        
        # Save cache info
        cache_info = {
            'num_train_samples': len(train_predictions),
            'num_val_samples': len(val_predictions),
            'feature_dim': train_features.shape[1] if len(train_features) > 0 else 0,
            'confidence_threshold': self.confidence_threshold,
            'checkpoint_path': str(self.checkpoint_path),
            'generated_at': str(torch.utils.data.get_worker_info() or "unknown")
        }
        
        with open(self.output_dir / "cache_info.json", 'w') as f:
            json.dump(cache_info, f, indent=2)
        
        print("Cache files saved successfully!")
        print(f"Train samples: {len(train_predictions)}")
        print(f"Val samples: {len(val_predictions)}")
        print(f"Feature dimension: {train_features.shape[1] if len(train_features) > 0 else 0}")
        
        # Print file sizes
        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.name}: {size_mb:.1f} MB")
    
    def generate_cache(self, max_train_images: Optional[int] = None, max_val_images: Optional[int] = None):
        """
        Generate complete cache from checkpoint.
        
        Args:
            max_train_images: Maximum training images to process (for testing)
            max_val_images: Maximum validation images to process (for testing)
        """
        print("="*80)
        print("GENERATING CACHE FROM FASTER R-CNN CHECKPOINT")
        print("="*80)
        
        # Setup model
        self.setup_model()
        
        # Register datasets
        self.register_coco_datasets()
        
        # Process training data
        print("\n" + "="*50)
        print("PROCESSING TRAINING DATA")
        print("="*50)
        
        train_predictions, train_labels = self.run_inference_on_dataset("coco_train", max_train_images)
        train_matched_preds, train_matched_labels = self.match_predictions_to_ground_truth(train_predictions, train_labels)
        train_features = self.extract_features(train_matched_preds)
        
        # Process validation data
        print("\n" + "="*50)
        print("PROCESSING VALIDATION DATA")
        print("="*50)
        
        val_predictions, val_labels = self.run_inference_on_dataset("coco_val", max_val_images)
        val_matched_preds, val_matched_labels = self.match_predictions_to_ground_truth(val_predictions, val_labels)
        val_features = self.extract_features(val_matched_preds)
        
        # Save cache
        print("\n" + "="*50)
        print("SAVING CACHE")
        print("="*50)
        
        self.save_cache(
            train_matched_preds, train_matched_labels,
            val_matched_preds, val_matched_labels,
            train_features, val_features
        )
        
        print("\n" + "="*80)
        print("CACHE GENERATION COMPLETED!")
        print("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate cache from Faster R-CNN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Faster R-CNN checkpoint")
    parser.add_argument("--coco-dir", type=str, required=True,
                        help="Path to COCO dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cache files")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--confidence-threshold", type=float, default=0.05,
                        help="Minimum confidence threshold for predictions")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Maximum training images to process (for testing)")
    parser.add_argument("--max-val-images", type=int, default=None,
                        help="Maximum validation images to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return 1
    
    if not os.path.exists(args.coco_dir):
        print(f"Error: COCO directory {args.coco_dir} does not exist")
        return 1
    
    # Create cache generator
    generator = CacheGenerator(
        checkpoint_path=args.checkpoint,
        coco_data_dir=args.coco_dir,
        output_dir=args.output_dir,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Generate cache
    try:
        generator.generate_cache(
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images
        )
        return 0
    except Exception as e:
        print(f"Error generating cache: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 