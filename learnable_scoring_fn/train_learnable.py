#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the learnable scoring function.

This script handles the training and evaluation of the learnable scoring function
for conformal prediction in object detection. It follows the same pattern as the
main.py file but is specifically focused on the learnable scoring function.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.insert(0, "/ssd_4TB/divake/conformal-od")
sys.path.insert(0, "/ssd_4TB/divake/conformal-od/detectron2")

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, DatasetCatalog

from control import classifier_sets
from data import data_loader
from model import model_loader
from util import util, io_file
from learnable_scoring_fn import LearnableConformal


def create_parser():
    """Create argument parser for training a learnable scoring function."""
    parser = argparse.ArgumentParser(description="Train a learnable scoring function for conformal prediction.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="cfg_learn_rank",
        required=False,
        help="Config file name to use for current run.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/ssd_4TB/divake/conformal-od/config/coco_val",
        required=False,
        help="Path to config file to use for current run.",
    )
    parser.add_argument(
        "--run_collect_pred",
        action="store_true",
        default=True,
        help="If run collect_predictions method (bool).",
    )
    parser.add_argument(
        "--no_run_collect_pred",
        action="store_false",
        dest="run_collect_pred",
        help="Do not run collect_predictions method.",
    )
    parser.add_argument(
        "--load_collect_pred",
        type=str,
        default=None,
        required=False,
        help="File name prefix from which to load pred info if not running collect_predictions",
    )
    parser.add_argument(
        "--save_file_pred",
        action="store_true",
        default=True,
        help="If save collect_predictions results to file (bool).",
    )
    parser.add_argument(
        "--no_save_file_pred",
        action="store_false",
        dest="save_file_pred",
        help="Do not save collect_predictions results to file.",
    )
    parser.add_argument(
        "--risk_control",
        type=str,
        default="learn_conf",
        required=False,
        help="Type of risk control/conformal approach to use.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        required=False,
        help="Alpha level for box coverage guarantee.",
    )
    parser.add_argument(
        "--label_set",
        type=str,
        default="class_threshold",
        required=False,
        help="Label set generation strategy to use.",
    )
    parser.add_argument(
        "--label_alpha",
        type=float,
        default=0.01,
        required=False,
        help="Alpha level for label coverage guarantee.",
    )
    parser.add_argument(
        "--run_risk_control",
        action="store_true",
        default=True,
        help="If run risk control procedure, i.e. controller.__call__ (bool).",
    )
    parser.add_argument(
        "--no_run_risk_control",
        action="store_false",
        dest="run_risk_control",
        help="Do not run risk control procedure.",
    )
    parser.add_argument(
        "--load_risk_control",
        type=str,
        default=None,
        required=False,
        help="File name prefix from which to load control info if not running risk control",
    )
    parser.add_argument(
        "--save_file_control",
        action="store_true",
        default=True,
        help="If save risk control procedure results to file (bool).",
    )
    parser.add_argument(
        "--no_save_file_control",
        action="store_false",
        dest="save_file_control",
        help="Do not save risk control procedure results to file.",
    )
    parser.add_argument(
        "--save_label_set",
        action="store_true",
        default=True,
        help="If save label set results to file (bool).",
    )
    parser.add_argument(
        "--no_save_label_set",
        action="store_false",
        dest="save_label_set",
        help="Do not save label set results to file.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        default=True,
        help="If run risk control evaluation, i.e. controller.evaluate (bool).",
    )
    parser.add_argument(
        "--no_run_eval",
        action="store_false",
        dest="run_eval",
        help="Do not run risk control evaluation.",
    )
    parser.add_argument(
        "--save_file_eval",
        action="store_true",
        default=True,
        help="If save results table to file (bool).",
    )
    parser.add_argument(
        "--no_save_file_eval",
        action="store_false",
        dest="save_file_eval",
        help="Do not save results table to file.",
    )
    parser.add_argument(
        "--file_name_prefix",
        type=str,
        default=None,
        required=False,
        help="File name prefix to save/load results under.",
    )
    parser.add_argument(
        "--file_name_suffix",
        type=str,
        default="_learn_rank_class",
        required=False,
        help="File name suffix to save/load results under.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="Device to run code on (cpu, cuda).",
    )
    return parser


def plot_training_metrics(metrics, output_dir):
    """Plot training metrics and save figures."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Extract metrics
    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    coverage_metrics = metrics["coverage_metrics"]
    best_epoch = metrics["best_epoch"]
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # Plot coverage metrics
    coverage = [m[0] for m in coverage_metrics]
    width = [m[1] for m in coverage_metrics]
    
    plt.figure(figsize=(10, 5))
    plt.plot(coverage, label='Coverage')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target Coverage (90%)')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Coverage')
    plt.title('Validation Coverage')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'coverage.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(width, label='Relative Width')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Relative Width')
    plt.title('Validation Prediction Interval Width')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'width.png'))
    plt.close()
    
    # Plot combined metrics
    fig, ax1 = plt.figure(figsize=(10, 5)), plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(coverage, 'b-', label='Coverage')
    ax1.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Coverage', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2.plot(width, 'g-', label='Width')
    ax2.set_ylabel('Relative Width', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.axvline(x=best_epoch, color='k', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Coverage and Width')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'coverage_width.png'))
    plt.close()


def main():
    """Main function to train and evaluate the learnable scoring function."""
    parser = create_parser()
    args = parser.parse_args()

    # Load config
    cfg = io_file.load_yaml(args.config_file, args.config_path, to_yacs=True)
    data_name = cfg.DATASETS.DATASET.NAME

    # Determine file naming and create experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.file_name_prefix is not None:
        file_name_prefix = args.file_name_prefix
    else:
        file_name_prefix = f"{args.risk_control}_{cfg.MODEL.ID}{args.file_name_suffix}"
    
    outdir = cfg.PROJECT.OUTPUT_DIR
    filedir = os.path.join(outdir, data_name, file_name_prefix)
    Path(filedir).mkdir(exist_ok=True, parents=True)

    # Set up logging
    logger = setup_logger(output=filedir)
    logger.info("Running training for learnable scoring function...")
    logger.info(f"Using config file '{args.config_file}'.")
    logger.info(f"Saving experiment files to '{filedir}'.")

    # Set seed & device
    util.set_seed(cfg.PROJECT.SEED, logger=logger)
    cfg, _ = util.set_device(cfg, args.device, logger=logger)

    # Register data with detectron2
    data_loader.d2_register_dataset(cfg, logger=logger)
    data_list = get_detection_dataset_dicts(
        data_name, filter_empty=cfg.DATASETS.DATASET.FILTER_EMPTY
    )
    dataloader = data_loader.d2_load_dataset_from_dict(
        data_list, cfg, None, logger=logger
    )
    metadata = MetadataCatalog.get(data_name).as_dict()
    nr_class = len(metadata["thing_classes"])
    nr_img = len(data_list)

    # Initialize risk control object
    logger.info(f"Initializing learnable scoring function controller...")
    controller = LearnableConformal(
        cfg, args, nr_class, filedir, log=False, logger=logger
    )

    # Initialize relevant DataCollector object
    controller.set_collector(nr_class, nr_img)
    
    # Load or collect model predictions
    if args.run_collect_pred:
        logger.info("Building and loading model...")
        cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
        model_loader.d2_load_model(cfg_model, model, logger=logger)
        
        logger.info("Collecting predictions...")
        img_list, ist_list = controller.collect_predictions(
            model, dataloader, verbose=False
        )
        
        if args.save_file_pred:
            logger.info("Saving predictions...")
            io_file.save_json(img_list, f"{file_name_prefix}_img_list", filedir)
            io_file.save_json(ist_list, f"{file_name_prefix}_ist_list", filedir)
    else:
        assert args.load_collect_pred is not None, "Need to load or collect predictions."
        logger.info(f"Loading predictions from '{args.load_collect_pred}'...")
        img_list = io_file.load_json(f"{args.load_collect_pred}_img_list", filedir)
        ist_list = io_file.load_json(f"{args.load_collect_pred}_ist_list", filedir)

    # Train the learnable scoring function
    logger.info("Training the learnable scoring function...")
    training_metrics = controller.train_model(img_list, ist_list)
    
    # Plot and save training metrics
    plot_dir = os.path.join(filedir, "plots")
    plot_training_metrics(training_metrics, plot_dir)
    
    # Save training metrics
    io_file.save_json(training_metrics, f"{file_name_prefix}_training_metrics", filedir)
    
    # Run risk control procedure
    if args.run_risk_control:
        logger.info("Running risk control procedure...")
        data, test_indices = controller(img_list, ist_list)
        
        if args.save_file_control:
            logger.info("Saving risk control data...")
            io_file.save_tensor(data, f"{file_name_prefix}_control", filedir)
            io_file.save_tensor(test_indices, f"{file_name_prefix}_test_idx", filedir)
    else:
        assert args.load_risk_control is not None, "Need to load or run risk control."
        logger.info(f"Loading risk control data from '{args.load_risk_control}'...")
        data = io_file.load_tensor(f"{args.load_risk_control}_control", filedir)
        test_indices = io_file.load_tensor(f"{args.load_risk_control}_test_idx", filedir)

    # Get label set data
    if controller.label_set_generator is not None:
        label_data = controller.label_set_generator.data
        box_set_data = controller.label_set_generator.box_set_data
        
        if args.save_label_set:
            logger.info("Saving label set data...")
            io_file.save_tensor(label_data, f"{file_name_prefix}_label", filedir)
            io_file.save_tensor(box_set_data, f"{file_name_prefix}_box_set", filedir)
    else:
        label_data = None
        box_set_data = None

    # Run evaluation
    if args.run_eval:
        logger.info("Evaluating performance...")
        results = controller.evaluate(
            data, label_data, box_set_data, metadata, filedir, 
            args.save_file_eval, args.load_collect_pred
        )
        
        logger.info("Results summary:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    main() 