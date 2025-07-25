#!/usr/bin/env python3
"""
Aggressive quantization to simulate real hardware effects
"""
import pickle
import numpy as np
import os
import time
from collections import OrderedDict

def aggressive_fp16_quantize(param):
    """More aggressive FP16 simulation with added noise"""
    # Convert to FP16 and back
    fp16_degraded = param.astype(np.float16).astype(np.float32)
    
    # Add quantization noise to simulate hardware effects
    # FP16 has 10-bit mantissa vs FP32's 23-bit
    noise_scale = np.abs(param).max() * 1e-4  # 0.01% noise
    noise = np.random.normal(0, noise_scale, param.shape).astype(np.float32)
    
    return fp16_degraded + noise

def aggressive_int8_quantize(param, percentile=99.0):
    """More aggressive INT8 quantization"""
    # Use lower percentile for more clipping
    scale = np.percentile(np.abs(param), percentile) / 127.0
    scale = max(scale, 1e-8)
    
    # Quantize with clipping
    quantized = np.round(param / scale).clip(-128, 127)
    
    # Add quantization noise
    # INT8 has much less precision - add more noise
    noise_scale = scale * 0.5  # Half a quantization level
    noise = np.random.uniform(-noise_scale, noise_scale, param.shape)
    
    # Dequantize with noise
    degraded = quantized * scale + noise
    
    return degraded.astype(np.float32)

def quantize_model_aggressive(model_path, output_path, quant_type='fp16'):
    """Create aggressively quantized model"""
    print(f"Creating aggressive {quant_type.upper()} model...")
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load checkpoint
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Create quantized checkpoint
    quant_checkpoint = checkpoint.copy()
    
    if 'model' in checkpoint:
        quant_model = OrderedDict()
        quant_count = 0
        skip_count = 0
        
        for name, param in checkpoint['model'].items():
            if isinstance(param, np.ndarray) and param.dtype == np.float32:
                # More aggressive: quantize almost everything
                skip_patterns = ['running_mean', 'running_var', 'num_batches_tracked']
                
                if any(skip in name for skip in skip_patterns):
                    quant_model[name] = param
                    skip_count += 1
                elif param.ndim >= 2:  # Weights (not biases)
                    if quant_type == 'fp16':
                        quant_model[name] = aggressive_fp16_quantize(param)
                    else:  # int8
                        # More aggressive for different layer types
                        if 'fpn' in name or 'lateral' in name:
                            # FPN layers more sensitive
                            quant_model[name] = aggressive_int8_quantize(param, percentile=99.5)
                        elif 'backbone' in name:
                            # Backbone can be more aggressive
                            quant_model[name] = aggressive_int8_quantize(param, percentile=98.0)
                        else:
                            # Everything else
                            quant_model[name] = aggressive_int8_quantize(param, percentile=99.0)
                    quant_count += 1
                else:  # Biases - also quantize but less aggressive
                    if quant_type == 'fp16':
                        quant_model[name] = param  # Keep biases in FP32 for FP16
                    else:
                        # Quantize biases too for INT8
                        quant_model[name] = aggressive_int8_quantize(param, percentile=99.9)
                        quant_count += 1
                    skip_count += 1
            else:
                quant_model[name] = param
        
        quant_checkpoint['model'] = quant_model
        print(f"  Quantized {quant_count} parameters")
        print(f"  Kept {skip_count} parameters with minimal/no quantization")
    
    # Save checkpoint
    print(f"  Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(quant_checkpoint, f)
    
    print(f"  Completed in {time.time() - start_time:.2f} seconds")

def verify_degradation(original_path, quantized_path, quant_type):
    """Verify degradation level"""
    print(f"\nVerifying {quant_type.upper()} degradation...")
    
    with open(original_path, 'rb') as f:
        orig = pickle.load(f)
    with open(quantized_path, 'rb') as f:
        quant = pickle.load(f)
    
    # Compare some weights
    total_error = 0
    count = 0
    max_rel_error = 0
    
    for name in list(orig['model'].keys())[:50]:  # Check first 50 weights
        if isinstance(orig['model'][name], np.ndarray) and orig['model'][name].ndim >= 2:
            orig_w = orig['model'][name]
            quant_w = quant['model'][name]
            
            # Relative error
            rel_error = np.abs(orig_w - quant_w) / (np.abs(orig_w) + 1e-8)
            mean_rel_error = rel_error.mean()
            max_rel_error = max(max_rel_error, rel_error.max())
            
            total_error += mean_rel_error
            count += 1
    
    avg_error = total_error / count if count > 0 else 0
    print(f"  Average relative error: {avg_error*100:.3f}%")
    print(f"  Max relative error: {max_rel_error*100:.1f}%")

if __name__ == "__main__":
    # Paths
    original = "/ssd_4TB/divake/conformal-od/checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pkl"
    fp16_aggressive = "/ssd_4TB/divake/conformal-od/checkpoints/faster_rcnn_X_101_32x8d_FPN_3x_fp16_aggressive.pkl"
    int8_aggressive = "/ssd_4TB/divake/conformal-od/checkpoints/faster_rcnn_X_101_32x8d_FPN_3x_int8_aggressive.pkl"
    
    print("Aggressive Quantization for Realistic Degradation\n" + "="*50)
    
    # Create aggressive versions
    quantize_model_aggressive(original, fp16_aggressive, 'fp16')
    quantize_model_aggressive(original, int8_aggressive, 'int8')
    
    # Verify degradation
    verify_degradation(original, fp16_aggressive, 'fp16')
    verify_degradation(original, int8_aggressive, 'int8')
    
    print("\nAggressive models created!")
    print("These should show more realistic degradation:") 
    print("- FP16: ~1-2% AP drop expected")
    print("- INT8: ~2-5% AP drop expected")