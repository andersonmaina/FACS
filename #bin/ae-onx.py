import torch
import torch.nn as nn
import os

# Define your AE class
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 2048)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Map categories to their .pth files
autoencoder_files = {
    "abdomen": "models/abdomen_autoencoder-0.0058.pth",
    "body": "models/body_autoencoder-0.0060.pth",
    "diencephalon": "models/diencephalon_autoencoder-0.0050.pth",
    "gsac": "models/gestation_sac_autoencoder-0.0044.pth",
    "head": "models/head_autoencoder-0.0077.pth",
    "lv": "models/lateral_ventricle_autoencoder-0.0045.pth",
    "mx": "models/maxilla_autoencoder-0.0054.pth",
    "mds": "models/mds_mandible_autoencoder-0.0039.pth",
    "mls": "models/mls_mandible_autoencoder-0.0047.pth",
    "nb": "models/nasal_bone_autoencoder-0.0026.pth",
    "ntaps": "models/ntaps_autoencoder-0.0032.pth",
    "rbp": "models/rhombencephalon_autoencoder-0.0044.pth",
    "thorax": "models/thorax_autoencoder-0.0058.pth",
    "test": "models/test.pth"
}

# Output folder for ONNX models
onnx_dir = "models/onnx"
os.makedirs(onnx_dir, exist_ok=True)

# Export each AE
for category, pth_path in autoencoder_files.items():
    print(f"Exporting {category}...")
    ae = AE()
    ae.load_state_dict(torch.load(pth_path, map_location="cpu"))
    ae.eval()

    # Dummy input matching AE input size (features from ResNet)
    dummy_input = torch.randn(1, 2048)

    # Export to ONNX
    onnx_path = os.path.join(onnx_dir, f"{category}_autoencoder.onnx")
    torch.onnx.export(
        ae,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["recon"]
    )
    print(f"Saved ONNX model to {onnx_path}")

print("All autoencoders exported successfully ✅")
