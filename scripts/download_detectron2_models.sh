#!/bin/bash
# Script to download Detectron2 pre-trained models for different performance tiers

# Set the base directory for checkpoints
CHECKPOINT_DIR="/ssd_4TB/divake/conformal-od/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "Downloading Detectron2 models for performance analysis..."

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

# Models already available (skip download)
echo "\n=== Existing Models ==="
echo "✓ Faster R-CNN X-101-32x8d FPN (top-tier, 43.0 AP) - faster_rcnn_X_101_32x8d_FPN_3x.pth"
echo "✓ Faster R-CNN R-50 FPN (mid-range, 40.2 AP) - faster_rcnn_R_50_FPN_3x.pkl"

# High Performance Models (AP 40-42)
echo "\n=== High Performance Models ==="
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" \
    "faster_rcnn_R_101_FPN_3x.pkl" \
    "Faster R-CNN R-101 FPN (42.0 AP)"

# One-Stage Models (Different Architecture)
echo "\n=== One-Stage Models ==="
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl" \
    "retinanet_R_50_FPN_3x.pkl" \
    "RetinaNet R-50 FPN (38.7 AP)"

# Anchor-Free Models
echo "\n=== Anchor-Free Models ==="
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fcos_R_50_FPN_1x/137257794/model_final_ae3d14.pkl" \
    "fcos_R_50_FPN_1x.pkl" \
    "FCOS R-50 FPN (39.2 AP)"

# Entry Level Models (Different Head)
echo "\n=== Entry Level Models ==="
download_model \
    "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl" \
    "faster_rcnn_R_50_C4_3x.pkl" \
    "Faster R-CNN R-50 C4 (38.4 AP)"

# Advanced Models (if needed)
echo "\n=== Optional Advanced Models ==="
echo "To download Cascade R-CNN (top accuracy, 44.3 AP):"
echo "wget -O $CHECKPOINT_DIR/cascade_rcnn_R_50_FPN_3x.pkl https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl"

echo "\n✅ Model download complete!"
echo "\n📊 Performance Summary:"
echo "- Top tier: X-101-32x8d FPN (43.0 AP) ✓"
echo "- High: R-101 FPN (42.0 AP)"
echo "- Mid: R-50 FPN (40.2 AP) ✓"
echo "- One-stage: RetinaNet R-50 FPN (38.7 AP)"
echo "- Anchor-free: FCOS R-50 FPN (39.2 AP)"
echo "- Entry: R-50 C4 (38.4 AP)"

echo "\n🚀 Ready to run experiments across different model architectures!"