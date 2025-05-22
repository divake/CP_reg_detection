#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting script

This script contains all the plotting code to generate different plots (listed under headers) 
used directly in the paper. It is overly verbose or inefficient at times (e.g. by recomputing 
predictions and bounding box intervals) and could be restructured or improved by leveraging 
precomputed results with filtering. However, it is very flexible and permits filtering results 
e.g. by ground truth matching, class name, set of classes etc.
"""

import sys
import os
import torch
import importlib
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.optimize import brentq
import itertools

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn as sns

# Add project paths
sys.path.insert(0, "/ssd_4TB/divake/conformal-od")
sys.path.insert(0, "/ssd_4TB/divake/conformal-od/detectron2")

from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, DatasetCatalog
from detectron2.data.detection_utils import annotations_to_instances
from detectron2.structures import Instances, Boxes
from detectron2.utils.logger import setup_logger

from control import std_conformal, ens_conformal, cqr_conformal, baseline_conformal, classifier_sets
from data import data_loader
from evaluation import results_table
from model import matching, model_loader, ensemble_boxes_wbf
from model.qr_head import QuantileROIHead
from plots import plot_util
from util import util, io_file

# Optionally import plot style if available
try:
    from plots.plot_style import *
except ImportError:
    print("Warning: Could not import plot_style module")

# scientific notation off for pytorch
torch.set_printoptions(sci_mode=False)

def save_fig(figname: str, **kwargs):
    """Save figure to file with given name"""
    plt.savefig(figname + ".png", format="png", **kwargs)
    print(f"Saved figure {figname}.")

def setup_model_and_data(rc="std", d="coco_val", device="cuda"):
    """Setup model and data based on configuration"""
    # simulate CLI with fixed parameters (see main.py for definitions)
    args_dict = {
        "config_file": f"cfg_{rc}_rank",
        "config_path": f"/ssd_4TB/divake/conformal-od/config/{d}",
        "run_collect_pred": False,
        "load_collect_pred": f"{rc}_conf_x101fpn_{rc}_rank_class",
        "save_file_pred": False,
        "risk_control": f"{rc}_conf",
        "alpha": 0.1,
        "label_set": "class_threshold",
        "label_alpha": 0.01,
        "run_risk_control": True,
        "load_risk_control": None,
        "save_file_control": True,
        "save_label_set": True,
        "run_eval": True,
        "save_file_eval": True,
        "file_name_prefix": None,
        "file_name_suffix": f"_{rc}_rank_class",
        "log_wandb": False,
        "device": device
    }
    args = argparse.Namespace(**args_dict)

    # main setup (see main.py)
    cfg = io_file.load_yaml(args.config_file, args.config_path, to_yacs=True)
    data_name = cfg.DATASETS.DATASET.NAME 
    cfg.MODEL.AP_EVAL = False

    if args.file_name_prefix is not None:
        file_name_prefix = args.file_name_prefix
    else:
        file_name_prefix = (f"{args.risk_control}_{cfg.MODEL.ID}{args.file_name_suffix}")

    outdir = cfg.PROJECT.OUTPUT_DIR 
    filedir = os.path.join(outdir, data_name, file_name_prefix)
    Path(filedir).mkdir(exist_ok=True, parents=True)

    logger = setup_logger(output=filedir)
    util.set_seed(cfg.PROJECT.SEED, logger=logger)
    cfg, _ = util.set_device(cfg, device, logger=logger)

    if not DatasetCatalog.__contains__(data_name):
        data_loader.d2_register_dataset(cfg, logger=logger)

    cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
    model_loader.d2_load_model(cfg_model, model, logger=logger)

    data_list = get_detection_dataset_dicts(data_name, filter_empty=cfg.DATASETS.DATASET.FILTER_EMPTY)
    dataloader = data_loader.d2_load_dataset_from_dict(data_list, cfg, cfg_model, logger=logger)
    metadata = MetadataCatalog.get(data_name).as_dict()
    nr_class = len(metadata["thing_classes"])

    # Initialize risk controller
    logger.info(f"Init risk control procedure with '{args.risk_control}'...")
    if args.risk_control == "std_conf":
        controller = std_conformal.StdConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    elif args.risk_control == "ens_conf":
        controller = ens_conformal.EnsConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    elif args.risk_control == "cqr_conf":
        controller = cqr_conformal.CQRConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    elif args.risk_control == "base_conf":
        controller = baseline_conformal.BaselineConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )

    # Load precomputed data
    control_data = io_file.load_tensor(f"{file_name_prefix}_control", filedir)
    test_indices = io_file.load_tensor(f"{file_name_prefix}_test_idx", filedir)
    label_data = io_file.load_tensor(f"{file_name_prefix}_label", filedir)

    # Get filenames and setup plot directory
    fnames = [data_list[i]["file_name"].split("/")[-1][:-4] for i in range(len(data_list))]
    channels = cfg.DATASETS.DATASET.CHANNELS
    plotdir = os.path.join("plots", data_name, file_name_prefix)
    Path(plotdir).mkdir(exist_ok=True, parents=True)

    # Get metric indices
    from evaluation.results_table import _idx_metrics as metr
    from evaluation.results_table import _idx_label_metrics as label_metr

    return (args, cfg, controller, model, data_list, dataloader, metadata, nr_class, 
            control_data, test_indices, label_data, fnames, channels, plotdir, 
            metr, label_metr, file_name_prefix, filedir)

def get_args(rc, d, device="cpu"):
    """Get command line arguments for a specific risk controller and dataset"""
    args_dict = {
        "config_file": f"cfg_{rc}_rank",
        "config_path": f"/ssd_4TB/divake/conformal-od/config/{d}",
        "run_collect_pred": False,
        "load_collect_pred": f"{rc}_conf_x101fpn_{rc}_rank_class",
        "save_file_pred": False,
        "risk_control": f"{rc}_conf",
        "alpha": 0.1,
        "label_set": "class_threshold",
        "label_alpha": 0.01,
        "run_risk_control": True,
        "load_risk_control": None,
        "save_file_control": True,
        "save_label_set": True,
        "run_eval": True,
        "save_file_eval": True,
        "file_name_prefix": None,
        "file_name_suffix": f"_{rc}_rank_class",
        "log_wandb": False,
        "device": device
    }
    args = argparse.Namespace(**args_dict)
    return args

def get_dirs(args, cfg):
    """Get directories for file saving"""
    if args.file_name_prefix is not None:
        file_name_prefix = args.file_name_prefix
    else:
        file_name_prefix = (f"{args.risk_control}_{cfg.MODEL.ID}{args.file_name_suffix}")
    outdir = cfg.PROJECT.OUTPUT_DIR
    data_name = cfg.DATASETS.DATASET.NAME
    filedir = os.path.join(outdir, data_name, file_name_prefix)
    Path(filedir).mkdir(exist_ok=True, parents=True)
    return file_name_prefix, outdir, filedir

def get_controller(args, cfg, nr_class, filedir, logger):
    """Get appropriate risk controller based on args"""
    logger.info(f"Init risk control procedure with '{args.risk_control}'...")
    if args.risk_control == "std_conf":
        controller = std_conformal.StdConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    elif args.risk_control == "ens_conf":
        controller = ens_conformal.EnsConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    elif args.risk_control == "cqr_conf":
        controller = cqr_conformal.CQRConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    elif args.risk_control == "base_conf":
        controller = baseline_conformal.BaselineConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    return controller

def get_loggy(plotdir_log, fname_log):
    """Create a logger for plot-specific logging"""
    loggy = logging.getLogger('loggy')
    loggy.setLevel(logging.DEBUG)
    loggy.propagate = 0
    file_handler = logging.FileHandler(os.path.join(plotdir_log, fname_log))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s|%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    loggy.addHandler(file_handler)
    loggy.addHandler(console_handler)
    return loggy

def update_log_path(loggy, new_path):
    """Update logger path"""
    while len(loggy.handlers) > 0:
        loggy.removeHandler(loggy.handlers[0])
    file_handler = logging.FileHandler(new_path)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s|%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    loggy.addHandler(file_handler)
    loggy.addHandler(console_handler)

def get_pred(args, controller, model, img, img_id, idx, filter_for_class, filter_for_set, class_name, set_name,
             set_idx, control_data, label_data, i, j, metr, label_metr, coco_classes, loggy, metadata):
    """Helper function to get predictions for comparison plots"""
    # prediction
    loggy.info("+++ Prediction procedure +++")
    
    # Special handling for ensemble model to avoid device mismatch
    if args.risk_control == "ens_conf":
        # For ensemble model, we need to handle device mismatches carefully
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        
        # Create a wrapper to handle device mismatches
        def wrapped_raw_prediction(controller, model, img):
            w, h = img["width"], img["height"]
            
            with torch.no_grad():
                pred_boxes, pred_classes, pred_scores, pred_score_all = [], [], [], []
                
                for m, ens_model in enumerate(controller.ensemble):
                    pred = ens_model([img])
                    ist = pred[0]["instances"]
                    # normalize boxes for weighted box fusion (wbf) - ensure matching devices
                    norm_tensor = torch.tensor([w, h, w, h], device=ist.pred_boxes.tensor.device)
                    box_norm = torch.div(ist.pred_boxes.tensor, norm_tensor)
                    pred_boxes.append(box_norm.tolist())
                    pred_classes.append(ist.pred_classes.tolist())
                    pred_scores.append(ist.scores.tolist())
                    pred_score_all.append(ist.scores_all.tolist())
                
                # wbf, modified to also return ensemble uncertainty
                boxes, scores, score_all, classes, unc = ensemble_boxes_wbf.weighted_boxes_fusion(
                    pred_boxes, pred_scores, pred_score_all, pred_classes
                )
                
                # Move results to the same device as the model
                box_unnorm = torch.tensor(boxes, device=device) * torch.tensor([w, h, w, h], device=device)
                unc_unnorm = torch.tensor(unc, device=device) * torch.tensor([w, h, w, h], device=device)
                # replace zero values with one, i.e., recover absolute residuals with std_dev = 1
                unc_unnorm = torch.where(unc_unnorm == 0, torch.tensor(1.0, device=device), unc_unnorm)
                
                ens_ist = Instances((h, w))
                ens_ist.set("pred_boxes", Boxes(box_unnorm))
                ens_ist.set("pred_classes", torch.tensor(classes, device=device).to(torch.int))
                ens_ist.set("scores", torch.tensor(scores, device=device))
                ens_ist.set("scores_all", torch.tensor(score_all, device=device))
                ens_ist.set("unc", unc_unnorm)
            
            return ens_ist
        
        # Use our wrapper instead of controller.raw_prediction
        pred = wrapped_raw_prediction(controller, model, img)
        loggy.info(f"Predicted for img {img_id} (idx {idx}) using wrapped {controller.__class__}")
    else:
        # Standard prediction for other models
        pred = controller.raw_prediction(model, img)
        loggy.info(f"Predicted for img {img_id} (idx {idx}) using {controller.__class__}")

    # filtering
    if filter_for_class:
        class_idx = metadata["thing_classes"].index(class_name)
        img["annotations"] = [anno for anno in img["annotations"] if anno["category_id"] == class_idx]
        loggy.info(f"Filtered for class '{class_name}' only.")
    elif filter_for_set:
        img["annotations"] = [anno for anno in img["annotations"] if anno["category_id"] in set_idx]
        loggy.info(f"Filtered for classes {set_name} only.")

    # matching
    gt = annotations_to_instances(img["annotations"], (img["height"], img["width"]))
    
    # Move everything to CPU for safe matching (the matching function uses numpy)
    gt_cpu = gt.to("cpu")
    pred_cpu = pred.to("cpu")

    (gt_box, pred_box, gt_class, pred_class, pred_score,
     pred_score_all, pred_logits_all, matches, _, pred_idx, _) = matching.matching(
        gt_cpu.gt_boxes, pred_cpu.pred_boxes, gt_cpu.gt_classes, pred_cpu.pred_classes, pred_cpu.scores, 
        pred_cpu.scores_all, None,
        controller.box_matching, controller.class_matching, controller.iou_thresh,
        return_idx=True
    )
    
    # Move results back to the original device
    device = pred.pred_boxes.device
    if gt_box is not None:
        gt_box = gt_box.to(device)
    if pred_box is not None:
        pred_box = pred_box.to(device)
    if gt_class is not None:
        gt_class = gt_class.to(device)
    if pred_class is not None:
        pred_class = pred_class.to(device)
    if pred_score is not None:
        pred_score = pred_score.to(device)
    if pred_score_all is not None:
        pred_score_all = pred_score_all.to(device)
    loggy.info(f"Performed matching using {controller.box_matching=} and {controller.class_matching=}.")
    loggy.info(f"Missed ground truth objects: {len(gt.gt_classes) - len(pred_idx)}/{len(gt.gt_classes)}.\n")

    # build matched prediction instance
    pred_match = Instances(pred.image_size)
    pred_match.set("pred_boxes", pred_box)
    pred_match.set("scores", pred_score)
    pred_match.set("pred_classes", pred_class)
    pred_match.set("pred_score_all", pred_score_all)
    pred_match = pred_match.to(device)  # Ensure it's on the right device

    if args.risk_control == "ens_conf":
        pred_match.set("unc", pred.unc[pred_idx])
    elif args.risk_control == "cqr_conf":
        pred_lower = pred.get(f"pred_boxes_{controller.q_str[controller.q_idx[0]]}")
        pred_upper = pred.get(f"pred_boxes_{controller.q_str[controller.q_idx[1]]}")
        pred_match.set("pred_lower", pred_lower[pred_idx])
        pred_match.set("pred_upper", pred_upper[pred_idx])

    # get quantiles for all classes, mean quantile over trials
    device = pred_match.pred_score_all.device
    box_quant_all = control_data[:, :, i:j, metr["quant"]].mean(dim=0).to(device)
    label_quant = label_data[:, :, label_metr["quant"]].mean(dim=0).to(device)
    # true box quantiles
    box_quant_true = box_quant_all[gt_class]

    # get label set
    label_set = controller.label_set_generator.get_pred_set(pred_match.pred_score_all, label_quant)
    label_set = controller.label_set_generator.handle_null_set(pred_match.pred_score_all, label_set)

    loggy.info("+++ Label set procedure +++")
    loggy.info(f"Using method '{args.label_set}'.")
    lab_gt, lab_pred, lab_set = [], [], []
    for i, labels in enumerate(label_set):
        l_gt = coco_classes[gt_class[i]]
        l_pred = coco_classes[pred_class[i]]
        l_set = [coco_classes[l] for l in torch.nonzero(labels, as_tuple=True)[0]]
        loggy.info(f"True class: '{l_gt}' | Pred class: '{l_pred}' | Label set: {l_set}")
        lab_gt.append(l_gt)
        lab_pred.append(l_pred)
        lab_set.append(l_set)

    # get box set quantiles
    loggy.info(f"Box quantile selection strategy: {controller.label_set_generator.box_set_strategy}.")
    # Make sure label_set and box_quant_all are on the same device
    label_set = label_set.to(device)
    box_quant, box_quant_idx = classifier_sets.box_set_strategy(
        label_set, box_quant_all, controller.label_set_generator.box_set_strategy)
    box_quant = box_quant.to(device)
    box_quant_idx = box_quant_idx.to(device)
    
    b = box_quant_idx.tolist()
    l_box_quant = [["class" for _ in range(4)] for _ in range(len(b))]
    for bi, bv in enumerate(b):
        for bj, bv2 in enumerate(bv):
            l_box_quant[bi][bj] = lab_set[bi][bv2] 
    loggy.info(f"Selected quantiles: {l_box_quant}")

    return gt, pred_match, box_quant, box_quant_true, lab_gt, lab_pred, lab_set

def plot_multi_method_comparison(img_name="000000054593", class_name="person", dataset="coco_val", device="cuda:1", to_file=False):
    """
    Plot prediction intervals for a specific image using multiple methods (std, ens, cqr)
    """
    # Setup for each method
    args_std = get_args("std", dataset, device)
    args_ens = get_args("ens", dataset, device)
    args_cqr = get_args("cqr", dataset, device)

    cfg_std = io_file.load_yaml(args_std.config_file, args_std.config_path, to_yacs=True)
    cfg_ens = io_file.load_yaml(args_ens.config_file, args_ens.config_path, to_yacs=True)
    cfg_cqr = io_file.load_yaml(args_cqr.config_file, args_cqr.config_path, to_yacs=True)

    # Use direct absolute path for CQR checkpoint
    cfg_cqr.MODEL.CHECKPOINT_PATH = "/ssd_4TB/divake/conformal-od/checkpoints/x101fpn_train_qr_5k_postprocess.pth"

    file_name_prefix_std, outdir_std, filedir_std = get_dirs(args_std, cfg_std)
    file_name_prefix_ens, outdir_ens, filedir_ens = get_dirs(args_ens, cfg_ens)
    file_name_prefix_cqr, outdir_cqr, filedir_cqr = get_dirs(args_cqr, cfg_cqr)

    logger = setup_logger(output=filedir_std)
    util.set_seed(cfg_std.PROJECT.SEED, logger=logger)

    if not DatasetCatalog.__contains__(dataset):
        data_loader.d2_register_dataset(cfg_std, logger=logger)

    # Load models
    cfg_model_std, model_std = model_loader.d2_build_model(cfg_std, logger=logger)
    model_loader.d2_load_model(cfg_model_std, model_std, logger=logger)
    
    cfg_model_ens, model_ens = model_loader.d2_build_model(cfg_ens, logger=logger)
    model_loader.d2_load_model(cfg_model_ens, model_ens, logger=logger)
    
    cfg_model_cqr, model_cqr = model_loader.d2_build_model(cfg_cqr, logger=logger)
    model_loader.d2_load_model(cfg_model_cqr, model_cqr, logger=logger)

    # Load dataset
    data_list = get_detection_dataset_dicts(dataset, filter_empty=cfg_std.DATASETS.DATASET.FILTER_EMPTY)
    dataloader = data_loader.d2_load_dataset_from_dict(data_list, cfg_std, cfg_model_std, logger=logger)
    metadata = MetadataCatalog.get(dataset).as_dict()
    nr_class = len(metadata["thing_classes"])

    # Get controllers
    controller_std = get_controller(args_std, cfg_std, nr_class, filedir_std, logger)
    controller_ens = get_controller(args_ens, cfg_ens, nr_class, filedir_ens, logger)
    controller_cqr = get_controller(args_cqr, cfg_cqr, nr_class, filedir_cqr, logger)

    # Load precomputed data
    control_data_std = io_file.load_tensor(f"{file_name_prefix_std}_control", filedir_std)
    test_indices_std = io_file.load_tensor(f"{file_name_prefix_std}_test_idx", filedir_std)
    label_data_std = io_file.load_tensor(f"{file_name_prefix_std}_label", filedir_std)

    control_data_ens = io_file.load_tensor(f"{file_name_prefix_ens}_control", filedir_ens)
    test_indices_ens = io_file.load_tensor(f"{file_name_prefix_ens}_test_idx", filedir_ens)
    label_data_ens = io_file.load_tensor(f"{file_name_prefix_ens}_label", filedir_ens)

    control_data_cqr = io_file.load_tensor(f"{file_name_prefix_cqr}_control", filedir_cqr)
    test_indices_cqr = io_file.load_tensor(f"{file_name_prefix_cqr}_test_idx", filedir_cqr)
    label_data_cqr = io_file.load_tensor(f"{file_name_prefix_cqr}_label", filedir_cqr)

    # Setup plotting directories
    channels = cfg_std.DATASETS.DATASET.CHANNELS
    plotdir_std = os.path.join("plots", dataset, file_name_prefix_std)
    plotdir_ens = os.path.join("plots", dataset, file_name_prefix_ens)
    plotdir_cqr = os.path.join("plots", dataset, file_name_prefix_cqr)
    plotdir_log = os.path.join("plots", dataset, "logs")
    
    Path(plotdir_std).mkdir(exist_ok=True, parents=True)
    Path(plotdir_ens).mkdir(exist_ok=True, parents=True)
    Path(plotdir_cqr).mkdir(exist_ok=True, parents=True)
    Path(plotdir_log).mkdir(exist_ok=True, parents=True)
    
    # Setup logger
    loggy = get_loggy(plotdir_log, "log.txt")

    # Get metric indices
    from evaluation.results_table import _idx_metrics as metr
    from evaluation.results_table import _idx_label_metrics as label_metr

    # Get COCO classes
    coco_classes = util.get_coco_classes()
    sel_coco_classes = util.get_selected_coco_classes()

    # Parameters for prediction
    i, j = 0, 4  # desired score indices
    filter_for_class = True if class_name else False
    filter_for_set = False
    set_name = []
    set_idx = []

    # Find the image index
    fnames = [data_list[i]["file_name"].split("/")[-1][:-4] for i in range(len(data_list))]
    target_idx = fnames.index(img_name)
    idx = torch.tensor([target_idx], device=device)

    # Get the image
    img = dataloader.dataset.__getitem__(idx.item())
    img_id = os.path.splitext(os.path.basename(img["file_name"]))[0]
    
    # Ensure models are on the correct device
    model_std.to(device)
    model_ens.to(device)
    model_cqr.to(device)

    # Setup log file
    fname_log = f"all_{args_std.label_set}_{class_name}_idx{idx.item()}_img{img_id}.log"
    update_log_path(loggy, os.path.join(plotdir_log, fname_log))
    
    # Move tensor data to desired device
    control_data_std = control_data_std.to(device)
    control_data_ens = control_data_ens.to(device)
    control_data_cqr = control_data_cqr.to(device)
    test_indices_std = test_indices_std.to(device)

    # Generate predictions for each method
    loggy.info(f"------ Method: {args_std.risk_control} ------")
    gt_std, pred_match_std, box_quant_std, box_quant_true_std, lab_gt_std, lab_pred_std, lab_set_std = get_pred(
        args_std, controller_std, model_std, img, img_id, idx, filter_for_class, filter_for_set, 
        class_name, set_name, set_idx, control_data_std, label_data_std, i, j, metr, label_metr, 
        coco_classes, loggy, metadata
    )

    loggy.info(f"\n------ Method: {args_ens.risk_control} ------")
    gt_ens, pred_match_ens, box_quant_ens, box_quant_true_ens, lab_gt_ens, lab_pred_ens, lab_set_ens = get_pred(
        args_ens, controller_ens, model_ens, img, img_id, idx, filter_for_class, filter_for_set, 
        class_name, set_name, set_idx, control_data_ens, label_data_ens, i, j, metr, label_metr, 
        coco_classes, loggy, metadata
    )

    loggy.info(f"\n------ Method: {args_cqr.risk_control} ------")
    gt_cqr, pred_match_cqr, box_quant_cqr, box_quant_true_cqr, lab_gt_cqr, lab_pred_cqr, lab_set_cqr = get_pred(
        args_cqr, controller_cqr, model_cqr, img, img_id, idx, filter_for_class, filter_for_set, 
        class_name, set_name, set_idx, control_data_cqr, label_data_cqr, i, j, metr, label_metr, 
        coco_classes, loggy, metadata
    )

    # Plot with label set quantiles
    cn = class_name.replace(" ", "") if class_name else "all"
    
    fname_std = f"{args_std.risk_control}_{args_std.label_set}_{cn}_idx{idx.item()}_img{img_id}.jpg"
    fname_ens = f"{args_ens.risk_control}_{args_ens.label_set}_{cn}_idx{idx.item()}_img{img_id}.jpg"
    fname_cqr = f"{args_cqr.risk_control}_{args_cqr.label_set}_{cn}_idx{idx.item()}_img{img_id}.jpg"

    print(f"FIG 1.1: Label set quant; {args_std.risk_control} - {args_std.label_set}\n")
    plot_util.d2_plot_pi(args_std.risk_control, img, gt_std.gt_boxes, pred_match_std, box_quant_std,
                        channels, draw_labels=[], 
                        colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                        lw=1.5, notebook=True, to_file=to_file,
                        filename=os.path.join(plotdir_std, fname_std),
                        label_gt=lab_gt_std, label_set=lab_set_std)

    print(f"FIG 1.2: Label set quant; {args_ens.risk_control} - {args_ens.label_set}\n")
    plot_util.d2_plot_pi(args_ens.risk_control, img, gt_ens.gt_boxes, pred_match_ens, box_quant_ens,
                        channels, draw_labels=[], 
                        colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                        lw=1.5, notebook=True, to_file=to_file,
                        filename=os.path.join(plotdir_ens, fname_ens),
                        label_gt=lab_gt_ens, label_set=lab_set_ens)

    print(f"FIG 1.3: Label set quant; {args_cqr.risk_control} - {args_cqr.label_set}\n")
    plot_util.d2_plot_pi(args_cqr.risk_control, img, gt_cqr.gt_boxes, pred_match_cqr, box_quant_cqr,
                        channels, draw_labels=[], 
                        colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                        lw=1.5, notebook=True, to_file=to_file,
                        filename=os.path.join(plotdir_cqr, fname_cqr),
                        label_gt=lab_gt_cqr, label_set=lab_set_cqr)
    
    # Optional: plot with oracle quantiles
    if to_file:
        fname_std = f"{args_std.risk_control}_oracle_{cn}_idx{idx.item()}_img{img_id}.jpg"
        fname_ens = f"{args_ens.risk_control}_oracle_{cn}_idx{idx.item()}_img{img_id}.jpg"
        fname_cqr = f"{args_cqr.risk_control}_oracle_{cn}_idx{idx.item()}_img{img_id}.jpg"

        print(f"FIG 2.1: Oracle; {args_std.risk_control}\n")
        plot_util.d2_plot_pi(args_std.risk_control, img, gt_std.gt_boxes, pred_match_std, box_quant_true_std,
                            channels, draw_labels=[], 
                            colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                            lw=1.5, notebook=True, to_file=to_file,
                            filename=os.path.join(plotdir_std, fname_std),
                            label_gt=lab_gt_std, label_set=lab_set_std)

        print(f"FIG 2.2: Oracle; {args_ens.risk_control}\n")
        plot_util.d2_plot_pi(args_ens.risk_control, img, gt_ens.gt_boxes, pred_match_ens, box_quant_true_ens,
                            channels, draw_labels=[], 
                            colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                            lw=1.5, notebook=True, to_file=to_file,
                            filename=os.path.join(plotdir_ens, fname_ens),
                            label_gt=lab_gt_ens, label_set=lab_set_ens)

        print(f"FIG 2.3: Oracle; {args_cqr.risk_control}\n")
        plot_util.d2_plot_pi(args_cqr.risk_control, img, gt_cqr.gt_boxes, pred_match_cqr, box_quant_true_cqr,
                            channels, draw_labels=[], 
                            colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                            lw=1.5, notebook=True, to_file=to_file,
                            filename=os.path.join(plotdir_cqr, fname_cqr),
                            label_gt=lab_gt_cqr, label_set=lab_set_cqr)

if __name__ == "__main__":
    # Use default parameters matching the notebook
    plot_multi_method_comparison(
        img_name="000000054593",
        class_name="person",
        dataset="coco_val",
        device="cuda:1",
        to_file=False
    )