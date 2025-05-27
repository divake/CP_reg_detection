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
# Configure matplotlib to avoid LaTeX issues
matplotlib.use('Agg')  # Use non-interactive backend

# Set all LaTeX-related parameters to False before importing pyplot
matplotlib.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'regular',
    'font.family': ['DejaVu Sans', 'sans-serif'],  # Remove Arial to avoid font warnings
    'axes.unicode_minus': False,
    'text.latex.preamble': '',
    'pgf.rcfonts': False,
    'pgf.texsystem': 'pdflatex',
    'svg.fonttype': 'none',
    'figure.max_open_warning': 0  # Disable the figure warning
})

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn as sns

# Ensure LaTeX is disabled
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'regular',
    'font.family': ['DejaVu Sans', 'sans-serif'],  # Remove Arial to avoid font warnings
    'axes.unicode_minus': False,
    'figure.max_open_warning': 0  # Disable the figure warning
})

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

def configure_matplotlib_no_latex():
    """Ensure matplotlib is configured to not use LaTeX"""
    import matplotlib
    matplotlib.rcParams.update({
        'text.usetex': False,
        'mathtext.default': 'regular',
        'font.family': ['DejaVu Sans', 'sans-serif'],  # Remove Arial to avoid font warnings
        'axes.unicode_minus': False,
        'text.latex.preamble': '',
        'pgf.rcfonts': False,
        'svg.fonttype': 'none',
        'figure.max_open_warning': 0  # Disable the figure warning
    })
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.default': 'regular',
        'font.family': ['DejaVu Sans', 'sans-serif'],  # Remove Arial to avoid font warnings
        'axes.unicode_minus': False,
        'figure.max_open_warning': 0  # Disable the figure warning
    })

# Ensure matplotlib configuration is applied
configure_matplotlib_no_latex()

def save_fig(figname: str, **kwargs):
    """Save figure to file with given name"""
    plt.savefig(figname + ".png", format="png", **kwargs)
    print(f"Saved figure {figname}.")
    plt.close()  # Close the figure to free memory

# ... existing code ... 