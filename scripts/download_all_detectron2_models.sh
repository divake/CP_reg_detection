#!/bin/bash
# Script to download ALL Detectron2 models for comprehensive analysis
# Organized by architecture type and performance characteristics

# Set the base directory for checkpoints
CHECKPOINT_DIR="/ssd_4TB/divake/conformal-od/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "========================================"
echo "Downloading ALL Detectron2 Models"
echo "========================================"

# Function to download model
download_model() {
    local url="$1"
    local filename="$2"
    local description="$3"
    
    if [ -f "$CHECKPOINT_DIR/$filename" ]; then
        echo "✓ $description already exists"
    else
        echo "⏬ Downloading $description..."
        wget -q --show-progress -O "$CHECKPOINT_DIR/$filename" "$url"
        echo "✓ Downloaded $description"
    fi
}

# ============================================
# EXISTING MODELS (Already downloaded)
# ============================================
echo "\n=== EXISTING MODELS ==="
echo "✓ faster_rcnn_X_101_32x8d_FPN_3x.pth - X-101 FPN (43.0 AP)"
echo "✓ faster_rcnn_R_50_FPN_3x.pkl - R-50 FPN (40.2 AP)"
echo "✓ faster_rcnn_R_50_C4_3x.pkl - R-50 C4 (38.4 AP)"

# ============================================
# TWO-STAGE DETECTORS
# ============================================
echo "\n=== TWO-STAGE DETECTORS ==="

# Faster R-CNN variants
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" \
    "faster_rcnn_R_101_FPN_3x.pkl" \
    "Faster R-CNN R-101 FPN (42.0 AP)"

download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl" \
    "faster_rcnn_R_101_C4_3x.pkl" \
    "Faster R-CNN R-101 C4 (41.1 AP)"

download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl" \
    "faster_rcnn_R_50_DC5_3x.pkl" \
    "Faster R-CNN R-50 DC5 (39.0 AP)"

download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl" \
    "faster_rcnn_R_101_DC5_3x.pkl" \
    "Faster R-CNN R-101 DC5 (40.6 AP)"

# Fast R-CNN (no RPN)
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/model_final_e5f7ce.pkl" \
    "fast_rcnn_R_50_FPN_1x.pkl" \
    "Fast R-CNN R-50 FPN (37.9 AP)"

# Cascade R-CNN
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl" \
    "cascade_mask_rcnn_R_50_FPN_3x.pkl" \
    "Cascade R-CNN R-50 FPN (42.1 AP)"

download_model \
    "https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_101_FPN_3x/138363239/model_final_0ca76c.pkl" \
    "cascade_mask_rcnn_R_101_FPN_3x.pkl" \
    "Cascade R-CNN R-101 FPN (42.8 AP)"

download_model \
    "https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/18131413/model_0039999_e76410.pkl" \
    "cascade_mask_rcnn_X_152_32x8d_FPN_IN5k.pkl" \
    "Cascade R-CNN X-152 FPN (45.6 AP)"

# ============================================
# ONE-STAGE DETECTORS
# ============================================
echo "\n=== ONE-STAGE DETECTORS ==="

# RetinaNet
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl" \
    "retinanet_R_50_FPN_3x.pkl" \
    "RetinaNet R-50 FPN (38.7 AP)"

download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl" \
    "retinanet_R_101_FPN_3x.pkl" \
    "RetinaNet R-101 FPN (40.4 AP)"

# FCOS (Anchor-free)
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fcos_R_50_FPN_1x/137257794/model_final_ae3d14.pkl" \
    "fcos_R_50_FPN_1x.pkl" \
    "FCOS R-50 FPN (39.2 AP)"

# ============================================
# SPECIALIZED DETECTORS
# ============================================
echo "\n=== SPECIALIZED DETECTORS ==="

# RPN only
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl" \
    "rpn_R_50_FPN_1x.pkl" \
    "RPN R-50 FPN (~32 AP)"

# TridentNet
echo "\n⚠️  TridentNet requires special setup from detectron2/projects/TridentNet"
echo "   Model URL: https://dl.fbaipublicfiles.com/detectron2/TridentNet/trident_fast_R_50_C4_3x/147219144/model_final_5e1732.pkl"

# PointRend
echo "\n⚠️  PointRend requires special setup from detectron2/projects/PointRend"
echo "   Model URL: https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"

# ============================================
# EFFICIENT BACKBONES
# ============================================
echo "\n=== EFFICIENT BACKBONES (RegNet) ==="

# RegNetX
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_regnetx_4gf_dds_fpn_1x/163100182/model_final_40b7d1.pkl" \
    "mask_rcnn_regnetx_4gf_fpn_1x.pkl" \
    "Mask R-CNN RegNetX-4GF FPN (41.5 AP)"

# RegNetY  
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_regnety_4gf_dds_fpn_1x/163100194/model_final_f7e3e9.pkl" \
    "mask_rcnn_regnety_4gf_fpn_1x.pkl" \
    "Mask R-CNN RegNetY-4GF FPN (42.0 AP)"

# ============================================
# SUMMARY
# ============================================
echo "\n========================================"
echo "DOWNLOAD SUMMARY"
echo "========================================"
echo "Two-Stage Detectors:"
echo "  - Faster R-CNN: R-50/R-101 with FPN/C4/DC5"
echo "  - Fast R-CNN: R-50 FPN (no RPN)"
echo "  - Cascade R-CNN: R-50/R-101/X-152 FPN"
echo "\nOne-Stage Detectors:"
echo "  - RetinaNet: R-50/R-101 FPN"
echo "  - FCOS: R-50 FPN (anchor-free)"
echo "\nSpecialized:"
echo "  - RPN only"
echo "  - TridentNet (requires project setup)"
echo "  - PointRend (requires project setup)"
echo "\nEfficient Backbones:"
echo "  - RegNetX-4GF"
echo "  - RegNetY-4GF"
echo "\n✅ Model download complete!"
echo "========================================"