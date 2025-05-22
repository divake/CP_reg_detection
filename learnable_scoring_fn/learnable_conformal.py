import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# Add project paths
sys.path.insert(0, "/ssd_4TB/divake/conformal-od")
sys.path.insert(0, "/ssd_4TB/divake/conformal-od/detectron2")

from detectron2.structures.boxes import Boxes
from detectron2.data.detection_utils import annotations_to_instances

from control.abstract_risk_control import RiskControl
from control.std_conformal import StdConformal
from data.data_collector import DataCollector, _default_dict_fields
from model import matching
from calibration import random_split, pred_intervals
from calibration.conformal_scores import abs_res, one_sided_res
from evaluation import metrics, results_table
from util.io_file import save_tensor, load_tensor, save_json, load_json

from .model import ScoringMLP, extract_features, CoverageLoss


class LearnableConformal(RiskControl):
    """
    This class implements a learnable scoring function for bounding box regression
    in conformal prediction for object detection.
    
    The key difference from StdConformal is that instead of using predefined
    nonconformity scores (like absolute residuals), we train a neural network
    to learn optimal nonconformity scores that result in better prediction intervals.
    """
    def __init__(
        self,
        cfg,
        args,
        nr_class: int,
        filedir: str,
        log=None,
        logger=None,
    ):
        self.cfg = cfg
        self.args = args
        self.nr_class = nr_class
        self.filedir = filedir
        self.logger = logger
        
        # Create checkpoints directory
        self.checkpoints_dir = os.path.join(filedir, "checkpoints")
        Path(self.checkpoints_dir).mkdir(exist_ok=True, parents=True)

        # Base model related attributes
        self.box_matching = cfg.MODEL.MATCHING.BOX
        self.class_matching = cfg.MODEL.MATCHING.CLASS
        self.iou_thresh = cfg.MODEL.MATCHING.IOU
        
        # Calibration related attributes
        self.calib_trials = cfg.CALIB.TRIALS
        self.calib_fraction = cfg.CALIB.FRACTION
        self.calib_alpha = args.alpha
        self.calib_box_corr = cfg.CALIB.BOX_CORRECTION
        
        # Label set related attributes
        self.label_set_generator = None
        if hasattr(args, "label_set") and args.label_set:
            from control.classifier_sets import get_label_set_generator
            self.label_set_generator = get_label_set_generator(
                args.label_set,
                args.label_alpha,
                self.calib_box_corr,
                nr_class,
                logger,
            )
        
        # Set up learnable model parameters
        self.input_dim = 5  # box coords + score
        self.hidden_dims = [128, 64]
        self.output_dim = 4  # One score per coordinate
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.max_epochs = 50
        self.batch_size = 64
        self.target_coverage = 1.0 - self.calib_alpha
        self.lambda_width = 0.1  # Balance between coverage and interval width
        
        # Set up metrics
        self.nr_metrics = metrics._nr_metrics
        
        # Define train/cal/val data split
        self.train_fraction = 0.5
        self.cal_fraction = 0.3
        self.val_fraction = 0.2
        
        # Initialize model
        self.scoring_model = ScoringMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        )
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(
            self.scoring_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = CoverageLoss(
            target_coverage=self.target_coverage,
            lambda_width=self.lambda_width
        )
        
        # Track best model
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        if logger:
            logger.info(
                f"""
                Initialized LearnableConformal with:
                - Input dimension: {self.input_dim}
                - Hidden dimensions: {self.hidden_dims}
                - Output dimension: {self.output_dim}
                - Learning rate: {self.learning_rate}
                - Target coverage: {self.target_coverage}
                - Max epochs: {self.max_epochs}
                - Train/Cal/Val split: {self.train_fraction}/{self.cal_fraction}/{self.val_fraction}
                """
            )

    def set_collector(self, nr_class: int, nr_img: int):
        """Set up data collector for storing predictions."""
        self.collector = LearnableConformalDataCollector(
            nr_class=nr_class,
            nr_img=nr_img,
            logger=self.logger,
            label_set_generator=self.label_set_generator
        )
        if self.logger:
            self.logger.info("Set collector.")

    def raw_prediction(self, model, img):
        """Get raw predictions from the base model."""
        with torch.no_grad():
            pred = model([img])
            return pred[0]["instances"]

    def collect_predictions(self, model, dataloader, verbose: bool = False):
        """
        Collect model predictions for all images in the dataset.
        This is the main data collection phase before training and calibration.
        """
        self.logger.info(
            f"""
            Collecting model predictions with matching:
            - Box matching: {self.box_matching}
            - Class matching: {self.class_matching}
            - IoU threshold: {self.iou_thresh}
            """
        )
        
        device = next(model.parameters()).device
        
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting predictions")):
            img = batch
            img_id = i
            
            # Get raw predictions from model
            pred = self.raw_prediction(model, img)
            
            # Create ground truth instances
            gt = annotations_to_instances(
                img["annotations"], (img["height"], img["width"])
            ).to(device)
            
            if len(gt) == 0 or len(pred) == 0:
                continue
                
            # Match predictions to ground truth
            (gt_box, pred_box, gt_class, pred_class, pred_score, 
             pred_score_all, pred_logits_all, matches, _) = matching.matching(
                gt.gt_boxes, pred.pred_boxes, gt.gt_classes, pred.pred_classes, 
                pred.scores, pred.scores_all, None,
                self.box_matching, self.class_matching, self.iou_thresh
            )
            
            # Skip if no matches
            if not matches:
                continue
                
            # Collect predictions
            self.collector(
                gt_box,
                gt_class,
                pred_box,
                pred_score,
                pred_score_all,
                pred_logits_all,
                img_id=img_id,
                verbose=verbose
            )
        
        return self.collector.img_list, self.collector.ist_list

    def train_model(self, img_list: list, ist_list: list):
        """
        Train the learnable scoring function model using collected data.
        
        Args:
            img_list: List of image indicators per class
            ist_list: List of instance data per class
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info("Starting model training...")
        
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scoring_model.to(device)
        
        # Prepare dataset for training
        all_features = []
        all_gt_coords = []
        all_pred_coords = []
        
        for c in range(self.nr_class):
            # Skip classes with no instances
            if len(ist_list[c]["gt_x0"]) == 0:
                continue
                
            # Get ground truth coordinates
            gt_coords = torch.tensor([
                ist_list[c]["gt_x0"],
                ist_list[c]["gt_y0"],
                ist_list[c]["gt_x1"],
                ist_list[c]["gt_y1"]
            ]).T
            
            # Get predicted coordinates
            pred_coords = torch.tensor([
                ist_list[c]["pred_x0"],
                ist_list[c]["pred_y0"],
                ist_list[c]["pred_x1"],
                ist_list[c]["pred_y1"]
            ]).T
            
            # Get prediction scores
            pred_scores = torch.tensor(ist_list[c]["pred_score"])
            
            # Extract features for model input
            features = extract_features(pred_coords, pred_scores)
            
            all_features.append(features)
            all_gt_coords.append(gt_coords)
            all_pred_coords.append(pred_coords)
        
        # Concatenate all data
        if all_features:
            all_features = torch.cat(all_features, dim=0).to(device)
            all_gt_coords = torch.cat(all_gt_coords, dim=0).to(device)
            all_pred_coords = torch.cat(all_pred_coords, dim=0).to(device)
            
            # Create dataset indices
            n_samples = all_features.shape[0]
            indices = torch.randperm(n_samples)
            
            # Split into train/cal/val
            train_size = int(n_samples * self.train_fraction)
            cal_size = int(n_samples * self.cal_fraction)
            
            train_indices = indices[:train_size]
            cal_indices = indices[train_size:train_size + cal_size]
            val_indices = indices[train_size + cal_size:]
            
            # Create datasets
            train_features = all_features[train_indices]
            train_gt_coords = all_gt_coords[train_indices]
            train_pred_coords = all_pred_coords[train_indices]
            
            cal_features = all_features[cal_indices]
            cal_gt_coords = all_gt_coords[cal_indices]
            cal_pred_coords = all_pred_coords[cal_indices]
            
            val_features = all_features[val_indices]
            val_gt_coords = all_gt_coords[val_indices]
            val_pred_coords = all_pred_coords[val_indices]
            
            self.logger.info(f"Dataset split: train={train_size}, cal={cal_size}, val={len(val_indices)}")
            
            # Training loop
            train_losses = []
            val_losses = []
            coverage_metrics = []
            
            for epoch in range(self.max_epochs):
                # Training phase
                self.scoring_model.train()
                epoch_losses = []
                
                # Process in batches
                for b in range(0, train_size, self.batch_size):
                    batch_end = min(b + self.batch_size, train_size)
                    batch_features = train_features[b:batch_end]
                    batch_gt_coords = train_gt_coords[b:batch_end]
                    batch_pred_coords = train_pred_coords[b:batch_end]
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    scores = self.scoring_model(batch_features)
                    
                    # Compute prediction intervals based on scores
                    pred_lower = batch_pred_coords - scores
                    pred_upper = batch_pred_coords + scores
                    
                    # Compute loss
                    loss = self.criterion(pred_lower, pred_upper, batch_gt_coords)
                    epoch_losses.append(loss.item())
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                
                avg_train_loss = sum(epoch_losses) / len(epoch_losses)
                train_losses.append(avg_train_loss)
                
                # Calibration phase - calculate tau based on calibration set
                self.scoring_model.eval()
                with torch.no_grad():
                    cal_scores = self.scoring_model(cal_features)
                    
                    # Calculate nonconformity scores for calibration set
                    cal_abs_errors = torch.abs(cal_gt_coords - cal_pred_coords)
                    cal_normalized_scores = cal_abs_errors / cal_scores
                    
                    # Compute quantiles (tau) for each coordinate
                    quantile = torch.tensor(1 - self.calib_alpha).to(device)
                    taus = torch.quantile(cal_normalized_scores, quantile, dim=0)
                    
                    # Validation phase - using current tau
                    val_scores = self.scoring_model(val_features)
                    
                    # Create prediction intervals using tau
                    val_pred_lower = val_pred_coords - val_scores * taus
                    val_pred_upper = val_pred_coords + val_scores * taus
                    
                    # Compute validation loss
                    val_loss = self.criterion(val_pred_lower, val_pred_upper, val_gt_coords)
                    val_losses.append(val_loss.item())
                    
                    # Compute coverage metric
                    in_interval = (val_gt_coords >= val_pred_lower) & (val_gt_coords <= val_pred_upper)
                    coverage = torch.mean(in_interval.float(), dim=0)
                    avg_coverage = coverage.mean().item()
                    avg_width = ((val_pred_upper - val_pred_lower).mean() / torch.abs(val_pred_coords).mean()).item()
                    coverage_metrics.append((avg_coverage, avg_width))
                    
                    # Check if this is the best model so far
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss.item()
                        self.best_epoch = epoch
                        self.save_model(epoch)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs}: "
                    f"Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"Coverage={avg_coverage:.4f}, "
                    f"Width={avg_width:.4f}, "
                    f"Tau={taus.cpu().numpy()}"
                )
            
            # Load the best model
            self.load_model(self.best_epoch)
            self.logger.info(f"Loaded best model from epoch {self.best_epoch+1}")
            
            return {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "coverage_metrics": coverage_metrics,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "final_tau": taus.cpu().numpy()
            }
        else:
            self.logger.warning("No features available for training")
            return {
                "status": "failed",
                "reason": "No features available for training"
            }

    def save_model(self, epoch):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.scoring_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_val_loss,
        }, os.path.join(self.checkpoints_dir, f"model_epoch_{epoch}.pt"))
        
        # Also save as best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.scoring_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_val_loss,
        }, os.path.join(self.checkpoints_dir, "best_model.pt"))

    def load_model(self, epoch=None):
        """Load model checkpoint."""
        if epoch is not None:
            checkpoint_path = os.path.join(self.checkpoints_dir, f"model_epoch_{epoch}.pt")
        else:
            checkpoint_path = os.path.join(self.checkpoints_dir, "best_model.pt")
            
        checkpoint = torch.load(checkpoint_path)
        self.scoring_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch']

    def __call__(self, img_list: list, ist_list: list):
        """
        Run the risk control procedure using the trained scoring function.
        
        Args:
            img_list: List of image indicators per class
            ist_list: List of instance data per class
            
        Returns:
            data: Tensor containing coordinate/score-wise information
            test_indices: Boolean tensor recording which images end up in the test set
        """
        self.logger.info(
            f"""
            Running risk control procedure for {self.calib_trials} trials...
            Calibration fraction: {self.calib_fraction}, alpha: {self.calib_alpha},
            box correction: {self.calib_box_corr}.
            """
        )
        
        # Tensors to store information in
        data = torch.zeros(
            size=(
                self.calib_trials,
                self.nr_class,
                self.collector.nr_scores,
                self.nr_metrics,
            ),
            dtype=torch.float32,
        )
        test_indices = torch.zeros(
            size=(self.calib_trials, self.nr_class, self.collector.nr_img),
            dtype=torch.bool,
        )
        
        # Collect label set information
        if self.label_set_generator is not None:
            self.label_set_generator.collect(
                img_list,
                ist_list,
                self.nr_class,
                self.nr_metrics,
                self.collector.nr_scores,
                self.collector.score_fields,
                self.collector.coord_fields,
            )
        
        # Initialize device
        device = next(self.scoring_model.parameters()).device
        
        # For each trial and class, compute nonconformity scores and quantiles
        for t in range(self.calib_trials):
            self.logger.info(f"Running trial {t+1}/{self.calib_trials}...")
            
            for c in range(self.nr_class):
                # Skip if no instances for this class
                if sum(img_list[c]) == 0:
                    continue
                
                # Get indices of images with this class
                img_idx = torch.tensor([i for i, has_class in enumerate(img_list[c]) if has_class])
                
                # Skip if too few images
                if len(img_idx) < 2:
                    continue
                
                # Random split into calibration and test sets
                calib_idx, test_idx = random_split(
                    img_idx, self.calib_fraction, seed=t * self.nr_class + c
                )
                
                # Record test indices
                for i in test_idx:
                    test_indices[t, c, i] = True
                
                # Prepare features for scoring model
                pred_coords = torch.tensor([
                    [ist_list[c]["pred_x0"][i] for i in range(len(ist_list[c]["pred_x0"])) if ist_list[c]["img_id"][i] in calib_idx],
                    [ist_list[c]["pred_y0"][i] for i in range(len(ist_list[c]["pred_y0"])) if ist_list[c]["img_id"][i] in calib_idx],
                    [ist_list[c]["pred_x1"][i] for i in range(len(ist_list[c]["pred_x1"])) if ist_list[c]["img_id"][i] in calib_idx],
                    [ist_list[c]["pred_y1"][i] for i in range(len(ist_list[c]["pred_y1"])) if ist_list[c]["img_id"][i] in calib_idx]
                ]).T.to(device)
                
                gt_coords = torch.tensor([
                    [ist_list[c]["gt_x0"][i] for i in range(len(ist_list[c]["gt_x0"])) if ist_list[c]["img_id"][i] in calib_idx],
                    [ist_list[c]["gt_y0"][i] for i in range(len(ist_list[c]["gt_y0"])) if ist_list[c]["img_id"][i] in calib_idx],
                    [ist_list[c]["gt_x1"][i] for i in range(len(ist_list[c]["gt_x1"])) if ist_list[c]["img_id"][i] in calib_idx],
                    [ist_list[c]["gt_y1"][i] for i in range(len(ist_list[c]["gt_y1"])) if ist_list[c]["img_id"][i] in calib_idx]
                ]).T.to(device)
                
                pred_scores = torch.tensor([
                    ist_list[c]["pred_score"][i] for i in range(len(ist_list[c]["pred_score"])) 
                    if ist_list[c]["img_id"][i] in calib_idx
                ]).to(device)
                
                # Extract features
                features = extract_features(pred_coords, pred_scores)
                
                # Get learnable scores
                with torch.no_grad():
                    learned_scores = self.scoring_model(features)
                
                # Calculate nonconformity scores
                abs_errors = torch.abs(gt_coords - pred_coords)
                nonconf_scores = abs_errors / learned_scores
                
                # Compute quantiles for each coordinate
                quantile = 1 + (1 - self.calib_alpha) * (len(calib_idx) + 1) / len(calib_idx) - 1
                quantile = min(max(quantile, 0), 1)
                
                # Store quantiles
                for i, field in enumerate(self.collector.score_fields):
                    q = torch.quantile(nonconf_scores[:, i], quantile)
                    data[t, c, i, metrics._idx_metrics["quant"]] = q
                
                # Compute and store coverage metrics on test set
                test_features = torch.tensor([
                    [
                        ist_list[c]["pred_x0"][i],
                        ist_list[c]["pred_y0"][i],
                        ist_list[c]["pred_x1"][i],
                        ist_list[c]["pred_y1"][i]
                    ]
                    for i in range(len(ist_list[c]["pred_x0"])) 
                    if ist_list[c]["img_id"][i] in test_idx
                ]).to(device)
                
                test_gt = torch.tensor([
                    [
                        ist_list[c]["gt_x0"][i],
                        ist_list[c]["gt_y0"][i],
                        ist_list[c]["gt_x1"][i],
                        ist_list[c]["gt_y1"][i]
                    ]
                    for i in range(len(ist_list[c]["gt_x0"])) 
                    if ist_list[c]["img_id"][i] in test_idx
                ]).to(device)
                
                test_pred = torch.tensor([
                    [
                        ist_list[c]["pred_x0"][i],
                        ist_list[c]["pred_y0"][i],
                        ist_list[c]["pred_x1"][i],
                        ist_list[c]["pred_y1"][i]
                    ]
                    for i in range(len(ist_list[c]["pred_x0"])) 
                    if ist_list[c]["img_id"][i] in test_idx
                ]).to(device)
                
                test_scores = torch.tensor([
                    ist_list[c]["pred_score"][i] for i in range(len(ist_list[c]["pred_score"])) 
                    if ist_list[c]["img_id"][i] in test_idx
                ]).to(device)
                
                if len(test_scores) > 0:
                    test_features = extract_features(test_pred, test_scores)
                    
                    with torch.no_grad():
                        test_learned_scores = self.scoring_model(test_features)
                    
                    # Compute prediction intervals
                    for i, field in enumerate(self.collector.score_fields):
                        # Get quantile for this field
                        q = data[t, c, i, metrics._idx_metrics["quant"]]
                        
                        # Create prediction intervals
                        lower = test_pred[:, i] - q * test_learned_scores[:, i]
                        upper = test_pred[:, i] + q * test_learned_scores[:, i]
                        
                        # Compute coverage and width
                        covered = (test_gt[:, i] >= lower) & (test_gt[:, i] <= upper)
                        coverage = covered.float().mean()
                        width = (upper - lower).mean()
                        rel_width = width / torch.abs(test_pred[:, i]).mean()
                        
                        # Store metrics
                        data[t, c, i, metrics._idx_metrics["cover"]] = coverage
                        data[t, c, i, metrics._idx_metrics["width"]] = width
                        data[t, c, i, metrics._idx_metrics["rel_width"]] = rel_width
        
        return data, test_indices

    def evaluate(
        self,
        data: torch.Tensor,
        label_data: torch.Tensor,
        box_set_data: torch.Tensor,
        metadata: dict,
        filedir: str,
        save_file: bool,
        load_collect_pred,
    ):
        """Evaluate the performance of the risk control procedure."""
        self.logger.info("Collecting and computing results...")
        
        # Call existing evaluation from StdConformal
        std_controller = StdConformal(
            self.cfg, self.args, self.nr_class, filedir, log=None, logger=self.logger
        )
        
        return std_controller.evaluate(
            data, label_data, box_set_data, metadata, filedir, save_file, load_collect_pred
        )


class LearnableConformalDataCollector(DataCollector):
    """
    Subclass of DataCollector for the LearnableConformal risk control procedure.
    """
    def __init__(
        self,
        nr_class: int,
        nr_img: int,
        dict_fields: list = [],
        logger=None,
        label_set_generator=None,
    ):
        if not dict_fields:
            dict_fields = _default_dict_fields.copy()
            self.coord_fields = ["x0", "y0", "x1", "y1"]
            # Conformal scores - same as StdConformalDataCollector for compatibility
            self.score_fields = [
                "abs_res_x0",
                "abs_res_y0",
                "abs_res_x1",
                "abs_res_y1",
                "one_sided_res_x0",
                "one_sided_res_y0",
                "one_sided_res_x1",
                "one_sided_res_y1",
            ]
            self.nr_scores = len(self.score_fields)
            dict_fields += self.score_fields
        super().__init__(nr_class, nr_img, dict_fields, logger, label_set_generator)

    def __call__(
        self,
        gt_box: Boxes,
        gt_class: torch.Tensor,
        pred_box: Boxes,
        pred_score: torch.Tensor,
        pred_score_all: torch.Tensor,
        pred_logits_all: torch.Tensor = None,
        img_id: int = None,
        verbose: bool = False,
    ):
        for c in torch.unique(gt_class).numpy():
            # img has instances of class
            self.img_list[c][img_id] = 1
            # indices for matching instances
            idx = torch.nonzero(gt_class == c, as_tuple=True)[0]
            # Add base infos
            super()._add_instances(
                c,
                img_id,
                idx,
                gt_box,
                pred_box,
                pred_score,
                pred_score_all,
                pred_logits_all,
            )

            # Add conformal scores - same as StdConformalDataCollector for compatibility
            self.ist_list[c]["abs_res_x0"] += abs_res(
                gt_box[idx].tensor[:, 0], pred_box[idx].tensor[:, 0]
            ).tolist()
            self.ist_list[c]["abs_res_y0"] += abs_res(
                gt_box[idx].tensor[:, 1], pred_box[idx].tensor[:, 1]
            ).tolist()
            self.ist_list[c]["abs_res_x1"] += abs_res(
                gt_box[idx].tensor[:, 2], pred_box[idx].tensor[:, 2]
            ).tolist()
            self.ist_list[c]["abs_res_y1"] += abs_res(
                gt_box[idx].tensor[:, 3], pred_box[idx].tensor[:, 3]
            ).tolist()

            self.ist_list[c]["one_sided_res_x0"] += one_sided_res(
                gt_box[idx].tensor[:, 0], pred_box[idx].tensor[:, 0], min=True
            ).tolist()
            self.ist_list[c]["one_sided_res_y0"] += one_sided_res(
                gt_box[idx].tensor[:, 1], pred_box[idx].tensor[:, 1], min=True
            ).tolist()
            self.ist_list[c]["one_sided_res_x1"] += one_sided_res(
                gt_box[idx].tensor[:, 2], pred_box[idx].tensor[:, 2], min=False
            ).tolist()
            self.ist_list[c]["one_sided_res_y1"] += one_sided_res(
                gt_box[idx].tensor[:, 3], pred_box[idx].tensor[:, 3], min=False
            ).tolist()

        if verbose:
            print(f"Added all instances for image {img_id}.") 