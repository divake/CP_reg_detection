#!/usr/bin/env python3
"""
Create FP16 and INT8 quantized versions of Detectron2 model
Maintains exact pickle format with quantization-induced degradation
"""
import pickle
import numpy as np
import os
import time
from collections import OrderedDict
import glob
import json
from PIL import Image

def should_quantize_fp16(key):
    """Determine if a parameter should be quantized to FP16"""
    # Keep biases and norm parameters in FP32
    keep_fp32 = ['bias', 'norm', 'running_mean', 'running_var']
    return not any(term in key for term in keep_fp32)

def should_quantize_int8(key):
    """Determine if a parameter should be quantized to INT8"""
    # Quantize backbone conv layers and roi_heads layers
    quantize_patterns = [
        'backbone.bottom_up.res2', 'backbone.bottom_up.res3', 
        'backbone.bottom_up.res4', 'backbone.bottom_up.res5',
        'roi_heads.box_head', 'roi_heads.res5'
    ]
    
    # Don't quantize these
    skip_patterns = [
        'bias', 'norm', 'running_mean', 'running_var',
        'box_predictor', 'rpn_head', 'fpn_lateral'
    ]
    
    # Check if should skip
    if any(skip in key for skip in skip_patterns):
        return False
    
    # Check if should quantize
    return any(pattern in key for pattern in quantize_patterns)

def quantize_to_fp16(model_path, output_path):
    """Convert model to FP16 and back to FP32 for degradation"""
    print(f"Creating FP16 degraded model...")
    start_time = time.time()
    
    # Load checkpoint
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Create FP16 degraded checkpoint
    fp16_checkpoint = checkpoint.copy()
    
    if 'model' in checkpoint:
        fp16_model = OrderedDict()
        fp16_count = 0
        fp32_count = 0
        
        for name, param in checkpoint['model'].items():
            if isinstance(param, np.ndarray) and param.dtype == np.float32:
                if should_quantize_fp16(name):
                    # Quantize to FP16 and back to FP32
                    fp16_model[name] = param.astype(np.float16).astype(np.float32)
                    fp16_count += 1
                else:
                    # Keep in FP32
                    fp16_model[name] = param
                    fp32_count += 1
            else:
                fp16_model[name] = param
        
        fp16_checkpoint['model'] = fp16_model
        print(f"  Quantized {fp16_count} parameters to FP16")
        print(f"  Kept {fp32_count} parameters in FP32")
    
    # Save checkpoint
    print(f"  Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(fp16_checkpoint, f)
    
    print(f"  Completed in {time.time() - start_time:.2f} seconds")
    return fp16_checkpoint

def load_calibration_images(image_dir, num_images=50):
    """Load calibration images for INT8 quantization"""
    print(f"Loading calibration images from {image_dir}")
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))[:num_images]
    
    images = []
    for path in image_paths[:10]:  # Use first 10 for speed
        img = Image.open(path).convert('RGB')
        img_array = np.array(img)
        images.append(img_array)
    
    print(f"  Loaded {len(images)} calibration images")
    return images

def calibrate_scale(weight, method='percentile', percentile=99.9):
    """Compute quantization scale for a weight tensor"""
    weight_np = weight if isinstance(weight, np.ndarray) else weight.numpy()
    
    if method == 'absmax':
        scale = np.abs(weight_np).max() / 127.0
    elif method == 'percentile':
        scale = np.percentile(np.abs(weight_np), percentile) / 127.0
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    # Avoid zero scale
    scale = max(scale, 1e-8)
    return scale

def quantize_to_int8(model_path, output_path, calibration_images):
    """Convert model to INT8 and back to FP32 for degradation"""
    print(f"Creating INT8 degraded model...")
    start_time = time.time()
    
    # Load checkpoint
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Create INT8 degraded checkpoint
    int8_checkpoint = checkpoint.copy()
    
    if 'model' in checkpoint:
        int8_model = OrderedDict()
        int8_count = 0
        fp32_count = 0
        
        for name, param in checkpoint['model'].items():
            if isinstance(param, np.ndarray) and param.dtype == np.float32:
                if should_quantize_int8(name) and param.ndim >= 2:  # Only quantize weights, not biases
                    # Calibrate scale
                    scale = calibrate_scale(param, method='percentile', percentile=99.9)
                    
                    # Quantize and dequantize
                    quantized = np.round(param / scale).clip(-128, 127)
                    degraded = quantized * scale
                    
                    int8_model[name] = degraded.astype(np.float32)
                    int8_count += 1
                else:
                    # Keep in FP32
                    int8_model[name] = param
                    fp32_count += 1
            else:
                int8_model[name] = param
        
        int8_checkpoint['model'] = int8_model
        print(f"  Quantized {int8_count} parameters to INT8")
        print(f"  Kept {fp32_count} parameters in FP32")
    
    # Save checkpoint
    print(f"  Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(int8_checkpoint, f)
    
    print(f"  Completed in {time.time() - start_time:.2f} seconds")
    return int8_checkpoint

def verify_quantized_model(model_path, model_type):
    """Verify the quantized model"""
    print(f"\nVerifying {model_type} model: {model_path}")
    
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Check file size
    size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"  File size: {size_gb:.3f} GB")
    
    # Check model structure
    if 'model' in checkpoint:
        print(f"  Model has {len(checkpoint['model'])} parameters")
        
        # Sample some weights to verify degradation
        print(f"  Sample weights (showing degradation):")
        count = 0
        for name, param in checkpoint['model'].items():
            if isinstance(param, np.ndarray) and param.ndim >= 2 and count < 3:
                # Check if values look degraded
                unique_values = len(np.unique(param))
                print(f"    {name}: shape={param.shape}, unique_values={unique_values}")
                count += 1

if __name__ == "__main__":
    # Paths
    input_model = "/ssd_4TB/divake/conformal-od/checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pkl"
    fp16_model = "/ssd_4TB/divake/conformal-od/checkpoints/faster_rcnn_X_101_32x8d_FPN_3x_fp16.pkl"
    int8_model = "/ssd_4TB/divake/conformal-od/checkpoints/faster_rcnn_X_101_32x8d_FPN_3x_int8.pkl"
    calibration_dir = "/ssd_4TB/divake/conformal-od/data/coco/val2017"
    
    print("Detectron2 Model Quantization\n" + "="*50)
    
    # Create FP16 model
    print("\n1. Creating FP16 degraded model")
    quantize_to_fp16(input_model, fp16_model)
    
    # Load calibration images
    print("\n2. Loading calibration data for INT8")
    calibration_images = load_calibration_images(calibration_dir, num_images=50)
    
    # Create INT8 model
    print("\n3. Creating INT8 degraded model")
    quantize_to_int8(input_model, int8_model, calibration_images)
    
    # Verify models
    print("\n4. Verification")
    verify_quantized_model(fp16_model, "FP16")
    verify_quantized_model(int8_model, "INT8")
    
    print("\nQuantization complete! Models saved as:")
    print(f"  - {fp16_model}")
    print(f"  - {int8_model}")