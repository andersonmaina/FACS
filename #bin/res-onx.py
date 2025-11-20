#!/usr/bin/env python3
"""
Export ResNet-50 (feature extractor) to ONNX.

- Loads torchvision ResNet-50 with pretrained weights
- Replaces the final FC layer with Identity so output = 2048-d feature vector
- Exports to ONNX with dynamic batch/height/width
- Verifies the torch output shape before export

Usage:
  python export_resnet50_features.py [out_path] [opset]

Defaults:
  out_path = models/onnx/resnet50_features.onnx
  opset    = 12
"""

import os
import sys
import torch
import torch.nn as nn

def build_resnet50_feature_extractor():
    # Handle both new (weights=...) and old (pretrained=True) torchvision APIs
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        from torchvision.models import resnet50
        model = resnet50(pretrained=True)

    # Remove final classifier -> output becomes 2048-d features
    model.fc = nn.Identity()
    model.eval()
    return model

def main(out_path: str = "models/resnet50_features.onnx", opset: int = 12):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    model = build_resnet50_feature_extractor()

    # Dummy input (N, C, H, W); ResNet supports variable H/W via AdaptiveAvgPool
    dummy = torch.randn(1, 3, 224, 224)

    # Sanity check: ensure we really get (1, 2048)
    with torch.no_grad():
        y = model(dummy)
    assert y.shape == (1, 2048), f"Unexpected output shape: {tuple(y.shape)} (expected (1, 2048))"

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={
            "input":    {0: "batch", 2: "height", 3: "width"},
            "features": {0: "batch"}
        },
    )

    print(f"✅ Exported ResNet-50 feature extractor to: {out_path}")
    print("   Output tensor name: 'features'  |  Shape: (batch, 2048)")
    print(f"   ONNX opset: {opset}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "models/resnet50_features.onnx"
    opset = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    main(out, opset)
