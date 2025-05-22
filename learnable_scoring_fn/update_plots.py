#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update plots.py to include the learnable scoring function in comparison plots.

This script modifies the plot_multi_method_comparison function in plots.py
to add support for the learnable scoring function alongside std, ens, and cqr.
"""

import os
import sys
import re
from pathlib import Path

# Add project paths
sys.path.insert(0, "/ssd_4TB/divake/conformal-od")
sys.path.insert(0, "/ssd_4TB/divake/conformal-od/detectron2")

# Define the patch for adding learnable scoring function to get_args
get_args_patch = """
def get_args(rc, d, device="cpu"):
    \"\"\"Get command line arguments for a specific risk controller and dataset\"\"\"
    args_dict = {
        "config_file": f"cfg_{rc}_rank",
        "config_path": f"/ssd_4TB/divake/conformal-od/config/{d}",
        "run_collect_pred": False,
        "load_collect_pred": f"{rc}_conf_x101fpn_{rc}_rank_class",
        "save_file_pred": False,
        "risk_control": f"{rc}_conf" if rc != "learn" else "learn_conf",
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
"""

# Define the patch for adding learnable scoring function to get_controller
get_controller_patch = '''
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
    elif args.risk_control == "learn_conf":
        from learnable_scoring_fn import LearnableConformal
        controller = LearnableConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    elif args.risk_control == "base_conf":
        controller = baseline_conformal.BaselineConformal(
            cfg, args, nr_class, filedir, log=None, logger=logger
        )
    return controller
'''

# Define the patch for updating plot_multi_method_comparison to include learnable scoring function
plot_multi_method_comparison_patch = '''
def plot_multi_method_comparison(img_name="000000054593", class_name="person", dataset="coco_val", device="cuda:1", to_file=True, methods=None):
    """
    Plot prediction intervals for a specific image using multiple methods (std, ens, cqr, learn)
    
    Args:
        img_name: Name of the image to plot
        class_name: Class to filter for, or None for all classes
        dataset: Dataset to use
        device: Device to run on
        to_file: Whether to save plots to file
        methods: List of methods to plot, or None for all methods
    """
    # Create output directory for saving plots
    output_plots_dir = "/ssd_4TB/divake/conformal-od/output/plots"
    Path(output_plots_dir).mkdir(exist_ok=True, parents=True)
    
    # Default methods
    if methods is None:
        methods = ["std", "ens", "cqr", "learn"]
    
    # Setup for each method
    controllers = {}
    args_dict = {}
    cfg_dict = {}
    file_name_prefix_dict = {}
    filedir_dict = {}
    outdir_dict = {}
    model_dict = {}
    control_data_dict = {}
    test_indices_dict = {}
    label_data_dict = {}
    plotdir_dict = {}
    
    logger = None
    
    for method in methods:
        args = get_args(method, dataset, device)
        args_dict[method] = args
        
        cfg = io_file.load_yaml(args.config_file, args.config_path, to_yacs=True)
        cfg_dict[method] = cfg
        
        # Special handling for CQR checkpoint
        if method == "cqr":
            cfg.MODEL.CHECKPOINT_PATH = "/ssd_4TB/divake/conformal-od/checkpoints/x101fpn_train_qr_5k_postprocess.pth"
        
        file_name_prefix, outdir, filedir = get_dirs(args, cfg)
        file_name_prefix_dict[method] = file_name_prefix
        outdir_dict[method] = outdir
        filedir_dict[method] = filedir
        
        if logger is None:
            logger = setup_logger(output=filedir)
            util.set_seed(cfg.PROJECT.SEED, logger=logger)
            
            if not DatasetCatalog.__contains__(dataset):
                data_loader.d2_register_dataset(cfg, logger=logger)
        
        # Load model
        cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
        model_loader.d2_load_model(cfg_model, model, logger=logger)
        model_dict[method] = model
        
        # Get controller
        controller = get_controller(args, cfg, nr_class, filedir, logger)
        controllers[method] = controller
        
        # Load precomputed data
        control_data = io_file.load_tensor(f"{file_name_prefix}_control", filedir)
        test_indices = io_file.load_tensor(f"{file_name_prefix}_test_idx", filedir)
        label_data = io_file.load_tensor(f"{file_name_prefix}_label", filedir)
        
        control_data_dict[method] = control_data
        test_indices_dict[method] = test_indices
        label_data_dict[method] = label_data
        
        # Setup plotting directories
        channels = cfg.DATASETS.DATASET.CHANNELS
        plotdir = os.path.join("plots", dataset, file_name_prefix)
        Path(plotdir).mkdir(exist_ok=True, parents=True)
        plotdir_dict[method] = plotdir
    
    # Common data setup
    data_list = get_detection_dataset_dicts(dataset, filter_empty=cfg_dict[methods[0]].DATASETS.DATASET.FILTER_EMPTY)
    dataloader = data_loader.d2_load_dataset_from_dict(data_list, cfg_dict[methods[0]], cfg_model, logger=logger)
    metadata = MetadataCatalog.get(dataset).as_dict()
    nr_class = len(metadata["thing_classes"])
    
    # Setup plotting directories
    plotdir_log = os.path.join("plots", dataset, "logs")
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
    for method in methods:
        model_dict[method].to(device)
    
    # Setup log file
    fname_log = f"all_comparison_{class_name}_idx{idx.item()}_img{img_id}.log"
    update_log_path(loggy, os.path.join(plotdir_log, fname_log))
    
    # Prediction results for each method
    gt_dict = {}
    pred_match_dict = {}
    box_quant_dict = {}
    box_quant_true_dict = {}
    lab_gt_dict = {}
    lab_pred_dict = {}
    lab_set_dict = {}
    
    # Move tensor data to desired device
    for method in methods:
        control_data_dict[method] = control_data_dict[method].to(device)
        test_indices_dict[method] = test_indices_dict[method].to(device)
    
    # Generate predictions for each method
    for method in methods:
        loggy.info(f"\\n------ Method: {args_dict[method].risk_control} ------")
        gt, pred_match, box_quant, box_quant_true, lab_gt, lab_pred, lab_set = get_pred(
            args_dict[method], controllers[method], model_dict[method], img, img_id, idx, 
            filter_for_class, filter_for_set, class_name, set_name, set_idx, 
            control_data_dict[method], label_data_dict[method], i, j, metr, label_metr, 
            coco_classes, loggy, metadata
        )
        
        gt_dict[method] = gt
        pred_match_dict[method] = pred_match
        box_quant_dict[method] = box_quant
        box_quant_true_dict[method] = box_quant_true
        lab_gt_dict[method] = lab_gt
        lab_pred_dict[method] = lab_pred
        lab_set_dict[method] = lab_set
    
    # Plot with label set quantiles
    cn = class_name.replace(" ", "") if class_name else "all"
    
    # Create filenames for output directory
    output_fnames = {}
    for method in methods:
        output_fnames[method] = {
            'labelset': os.path.join(output_plots_dir, f"{args_dict[method].risk_control}_{args_dict[method].label_set}_{cn}_img{img_id}.jpg"),
            'oracle': os.path.join(output_plots_dir, f"{args_dict[method].risk_control}_oracle_{cn}_img{img_id}.jpg")
        }
    
    # Plot label set quantiles for each method
    for method in methods:
        print(f"FIG: Label set quant; {args_dict[method].risk_control} - {args_dict[method].label_set}\\n")
        plot_util.d2_plot_pi(args_dict[method].risk_control, img, gt_dict[method].gt_boxes, 
                            pred_match_dict[method], box_quant_dict[method],
                            cfg_dict[method].DATASETS.DATASET.CHANNELS, draw_labels=[], 
                            colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                            lw=1.5, notebook=True, to_file=to_file,
                            filename=output_fnames[method]['labelset'],
                            label_gt=lab_gt_dict[method], label_set=lab_set_dict[method])
    
    # Optional: plot with oracle quantiles
    if to_file:
        for method in methods:
            print(f"FIG: Oracle; {args_dict[method].risk_control}\\n")
            plot_util.d2_plot_pi(args_dict[method].risk_control, img, gt_dict[method].gt_boxes, 
                                pred_match_dict[method], box_quant_true_dict[method],
                                cfg_dict[method].DATASETS.DATASET.CHANNELS, draw_labels=[], 
                                colors=["red", "green", "palegreen"], alpha=[1.0, 0.6, 0.4],
                                lw=1.5, notebook=True, to_file=to_file,
                                filename=output_fnames[method]['oracle'],
                                label_gt=lab_gt_dict[method], label_set=lab_set_dict[method])
    
    # Print the output paths
    print("\\nSaved plots to:")
    for method in methods:
        print(f"  {method.capitalize()} model: {output_fnames[method]['labelset']}")
    if to_file:
        print("Oracle versions:")
        for method in methods:
            print(f"  {method.capitalize()} model: {output_fnames[method]['oracle']}")
'''

def update_plot_script():
    """Update the plots.py script to include the learnable scoring function."""
    # Path to the plots.py file
    plots_py_path = "/ssd_4TB/divake/conformal-od/plots/plots.py"
    
    # Backup the original file
    backup_path = "/ssd_4TB/divake/conformal-od/plots/plots.py.bak"
    if not os.path.exists(backup_path):
        with open(plots_py_path, 'r') as f_in, open(backup_path, 'w') as f_out:
            f_out.write(f_in.read())
        print(f"Created backup of plots.py at {backup_path}")
    
    # Read the current file
    with open(plots_py_path, 'r') as f:
        content = f.read()
    
    # Update the get_args function
    get_args_pattern = r"def get_args\(rc, d, device=\"cpu\"\):.*?return args"
    get_args_replacement = get_args_patch.strip()
    content = re.sub(get_args_pattern, get_args_replacement, content, flags=re.DOTALL)
    
    # Update the get_controller function
    get_controller_pattern = r"def get_controller\(args, cfg, nr_class, filedir, logger\):.*?return controller"
    get_controller_replacement = get_controller_patch.strip()
    content = re.sub(get_controller_pattern, get_controller_replacement, content, flags=re.DOTALL)
    
    # Update the plot_multi_method_comparison function
    plot_comparison_pattern = r"def plot_multi_method_comparison\(.*?\).*?if __name__ == \"__main__\":"
    plot_comparison_replacement = plot_multi_method_comparison_patch.strip() + "\n\nif __name__ == \"__main__\":"
    content = re.sub(plot_comparison_pattern, plot_comparison_replacement, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(plots_py_path, 'w') as f:
        f.write(content)
    
    print(f"Updated plots.py to include learnable scoring function support")

def update_imports():
    """Add required imports to plots.py."""
    plots_py_path = "/ssd_4TB/divake/conformal-od/plots/plots.py"
    
    with open(plots_py_path, 'r') as f:
        content = f.read()
    
    # Add import for LearnableConformal if it's not already there
    if "from learnable_scoring_fn import LearnableConformal" not in content:
        # Find the import section
        import_section_end = content.find("# scientific notation off for pytorch")
        if import_section_end != -1:
            # Add the new import just before the end of the import section
            new_content = content[:import_section_end] + "# Import learnable scoring function\n# (uncomment when needed)\n# from learnable_scoring_fn import LearnableConformal\n\n" + content[import_section_end:]
            
            # Write the updated content back to the file
            with open(plots_py_path, 'w') as f:
                f.write(new_content)
            
            print("Added import for LearnableConformal to plots.py")
        else:
            print("Could not find appropriate location to add import")

if __name__ == "__main__":
    print("Updating plots.py to add support for learnable scoring function...")
    update_plot_script()
    update_imports()
    print("Done!") 