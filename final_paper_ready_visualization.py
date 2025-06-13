#!/usr/bin/env python
"""
Final paper-ready visualization for symmetric adaptive conformal prediction.
Creates visualizations exactly matching the plots.py style with:
- Red prediction boxes
- Green inner and outer interval boxes  
- Light green shaded regions between intervals
- Confidence scores
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from pycocotools.coco import COCO

# Add paths
sys.path.append("/ssd_4TB/divake/conformal-od")
sys.path.append("/ssd_4TB/divake/conformal-od/detectron2")

# Import detectron2 components
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances

# Import our modules
from learnable_scoring_fn.core_symmetric.models.symmetric_mlp import SymmetricAdaptiveMLP
from util import util
from plots import plot_util
from plots.visualizer import VisualizerFill
from calibration import pred_intervals


def load_symmetric_model(experiment_dir):
    """Load the symmetric adaptive model."""
    model_path = Path(experiment_dir) / "models" / "best_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model_config = checkpoint.get('model_config', {})
    model = SymmetricAdaptiveMLP(
        input_dim=model_config.get('input_dim', 17),
        hidden_dims=model_config.get('hidden_dims', [128, 128]),
        activation=model_config.get('activation', 'elu'),
        dropout_rate=model_config.get('dropout_rate', 0.1),
        use_batch_norm=model_config.get('use_batch_norm', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tau = checkpoint.get('tau', 1.0)
    return model, tau


def extract_features_from_predictions(boxes, scores, img_shape):
    """Extract features for symmetric model from predictions."""
    n_boxes = len(boxes)
    if n_boxes == 0:
        return torch.zeros(0, 17)
    
    features = torch.zeros(n_boxes, 17)
    img_h, img_w = img_shape[:2]
    
    for i in range(n_boxes):
        box = boxes[i]
        score = scores[i]
        
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        # Same feature extraction as training
        features[i, 0] = x1 / img_w
        features[i, 1] = y1 / img_h
        features[i, 2] = x2 / img_w
        features[i, 3] = y2 / img_h
        features[i, 4] = w / img_w
        features[i, 5] = h / img_h
        features[i, 6] = (w * h) / (img_w * img_h)
        features[i, 7] = w / (h + 1e-6)
        features[i, 8] = float(score)
        features[i, 9] = (x1 + x2) / 2 / img_w
        features[i, 10] = (y1 + y2) / 2 / img_h
        features[i, 11] = abs((x1 + x2) / 2 - img_w / 2) / img_w
        features[i, 12] = abs((y1 + y2) / 2 - img_h / 2) / img_h
        features[i, 13] = x1 / img_w
        features[i, 14] = y1 / img_h
        features[i, 15] = (img_w - x2) / img_w
        features[i, 16] = (img_h - y2) / img_h
    
    return features


def create_final_visualization(image_path, predictor, symmetric_model, tau, 
                              output_path, class_filter=None, score_threshold=0.3):
    """Create final paper-ready visualization exactly matching plots.py style."""
    
    # Load image
    img_cv2 = cv2.imread(image_path)
    img_name = Path(image_path).stem
    img_h, img_w = img_cv2.shape[:2]
    
    # Get predictions from base model
    print(f"Getting predictions for {img_name}...")
    outputs = predictor(img_cv2)
    
    # Extract predictions
    instances = outputs["instances"].to("cpu")
    pred_boxes_tensor = instances.pred_boxes.tensor
    pred_scores = instances.scores
    pred_classes = instances.pred_classes
    
    # Load COCO metadata
    coco_ann_file = "/ssd_4TB/divake/conformal-od/data/coco/annotations/instances_val2017.json"
    coco = COCO(coco_ann_file)
    
    # Get image ID and ground truth
    img_id = int(img_name.lstrip('0'))
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # Get class names from COCO
    class_names = util.get_coco_classes()
    
    # Filter predictions by score
    high_conf_mask = pred_scores >= score_threshold
    filtered_instances = Instances((img_h, img_w))
    filtered_instances.pred_boxes = Boxes(pred_boxes_tensor[high_conf_mask])
    filtered_instances.scores = pred_scores[high_conf_mask]
    filtered_instances.pred_classes = pred_classes[high_conf_mask]
    
    # Filter by class if requested
    if class_filter:
        class_idx = class_names.index(class_filter)
        class_mask = filtered_instances.pred_classes == class_idx
        
        final_instances = Instances((img_h, img_w))
        final_instances.pred_boxes = Boxes(filtered_instances.pred_boxes.tensor[class_mask])
        final_instances.scores = filtered_instances.scores[class_mask]
        final_instances.pred_classes = filtered_instances.pred_classes[class_mask]
    else:
        final_instances = filtered_instances
    
    print(f"Found {len(final_instances)} predictions after filtering")
    
    # Extract features for symmetric model
    pred_boxes_np = final_instances.pred_boxes.tensor.numpy()
    pred_scores_np = final_instances.scores.numpy()
    features = extract_features_from_predictions(pred_boxes_np, pred_scores_np, img_cv2.shape)
    
    # Get symmetric intervals from our model
    if len(features) > 0:
        with torch.no_grad():
            predicted_widths = symmetric_model(features)
            # For symmetric intervals, we use the same width for all 4 sides
            # Convert to the format expected by std_conf (single value per box)
            symmetric_widths = predicted_widths.mean(dim=1, keepdim=True) * tau
            # Repeat for all 4 coordinates
            quant_values = symmetric_widths.expand(-1, 4)
    else:
        quant_values = torch.zeros((0, 4))
    
    # Get ground truth boxes
    gt_boxes = []
    for ann in anns:
        cat_name = coco.loadCats(ann['category_id'])[0]['name']
        if class_filter and cat_name != class_filter:
            continue
        x, y, w, h = ann['bbox']
        gt_boxes.append([x, y, x+w, y+h])
    
    gt_boxes_tensor = Boxes(torch.tensor(gt_boxes, dtype=torch.float32)) if gt_boxes else Boxes(torch.zeros((0, 4)))
    
    # Prepare image dict for plot_util
    img_dict = {
        "file_name": image_path,
        "height": img_h,
        "width": img_w,
        "image": torch.from_numpy(img_cv2.transpose(2, 0, 1))
    }
    
    # Use plot_util.d2_plot_pi to create the visualization
    # This will use the same style as plots.py
    cn = class_filter.replace(" ", "") if class_filter else "all"
    
    plot_util.d2_plot_pi(
        risk_control="std_conf",  # Use std_conf style for symmetric intervals
        image=img_dict,
        gt_box=gt_boxes_tensor,
        pred=final_instances,
        quant=quant_values,
        channel_order="BGR",
        draw_labels=[],  # No labels for cleaner look
        colors=["red", "green", "palegreen"],  # red for pred, green for intervals, palegreen for fill
        alpha=[1.0, 0.6, 0.4],  # alpha for pred, intervals, fill
        lw=1.5,
        notebook=False,
        to_file=True,
        filename=output_path,
        label_gt=None,
        label_set=None
    )
    
    return len(final_instances), len(gt_boxes)


def main():
    """Main function."""
    # Setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load symmetric model
    print("\nLoading symmetric adaptive model...")
    experiment_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric/symmetric_adaptive_20250612_081616"
    symmetric_model, tau = load_symmetric_model(experiment_dir)
    print(f"Loaded model with tau={tau:.4f}")
    
    # Setup detectron2 predictor
    cfg = get_cfg()
    cfg.merge_from_file("/ssd_4TB/divake/conformal-od/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = device
    
    predictor = DefaultPredictor(cfg)
    
    # Create visualization for baseball image (person only)
    image_path = "/ssd_4TB/divake/conformal-od/data/coco/val2017/000000054593.jpg"
    output_path = "symmetric_adaptive_paper_figure.jpg"
    
    print("\nCreating paper-ready visualization...")
    n_pred, n_gt = create_final_visualization(
        image_path,
        predictor,
        symmetric_model,
        tau,
        output_path,
        class_filter='person',
        score_threshold=0.3
    )
    
    print(f"\n{'='*60}")
    print("PAPER-READY VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Image: {Path(image_path).name}")
    print(f"Ground truth objects: {n_gt}")
    print(f"Predictions: {n_pred}")
    print(f"τ (tau): {tau:.2f}")
    print(f"Method: Symmetric Size-Aware Adaptive Conformal Prediction")
    print(f"\nOutput saved as: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()