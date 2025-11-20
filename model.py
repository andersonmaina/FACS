import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

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

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval()

autoencoder = AE()

def model_option(view, category):
    if view == "crl":
        if category == "abdomen":
            autoencoder.load_state_dict(torch.load('models/CRL/abdomen_autoencoder-0.0058.pth'))
        elif category == "body":
            autoencoder.load_state_dict(torch.load('models/CRL/body_autoencoder-0.0060.pth'))
        elif category == "diencephalon":
            autoencoder.load_state_dict(torch.load('models/CRL/diencephalon_autoencoder-0.0050.pth'))
        elif category == "gsac":
            autoencoder.load_state_dict(torch.load('models/CRL/gestation_sac_autoencoder-0.0044.pth'))
        elif category == "head":
            autoencoder.load_state_dict(torch.load('models/CRL/head_autoencoder-0.0077.pth'))
        elif category == "lv":
            autoencoder.load_state_dict(torch.load('models/CRL/lateral_ventricle_autoencoder-0.0045.pth'))
        elif category == "mx":
            autoencoder.load_state_dict(torch.load('models/CRL/maxilla_autoencoder-0.0054.pth'))
        elif category == "mds":
            autoencoder.load_state_dict(torch.load('models/CRL/mds_mandible_autoencoder-0.0039.pth'))
        elif category == "mls":
            autoencoder.load_state_dict(torch.load('models/CRL/mls_mandible_autoencoder-0.0047.pth'))
        elif category == "nb":
            autoencoder.load_state_dict(torch.load('models/CRL/nasal_bone_autoencoder-0.0026.pth'))
        elif category == "ntaps":
            autoencoder.load_state_dict(torch.load('models/CRL/ntaps_autoencoder-0.0032.pth'))
        elif category == "rbp":
            autoencoder.load_state_dict(torch.load('models/CRL/rhombencephalon_autoencoder-0.0044.pth'))
        elif category == "thorax":
            autoencoder.load_state_dict(torch.load('models/CRL/thorax_autoencoder-0.0058.pth'))
    elif view == "nt":
        if category == "abdomen":
            autoencoder.load_state_dict(torch.load('models/NT/abdomen_autoencoder-0.0050.pth'))
        elif category == "nuchal_translucency":
            autoencoder.load_state_dict(torch.load('models/NT/body_autoencoder-0.0064.pth'))
        elif category == "diencephalon":
            autoencoder.load_state_dict(torch.load('models/NT/diencephalon_autoencoder-0.0050.pth'))
        elif category == "head":
            autoencoder.load_state_dict(torch.load('models/NT/head_autoencoder-0.0067.pth'))
        elif category == "lv":
            autoencoder.load_state_dict(torch.load('models/NT/lateral_ventricle_autoencoder-0.0041.pth'))
        elif category == "mx":
            autoencoder.load_state_dict(torch.load('models/NT/maxilla_autoencoder-0.0038.pth'))
        elif category == "mds":
            autoencoder.load_state_dict(torch.load('models/NT/mds_mandible_autoencoder-0.0030.pth'))
        elif category == "mls":
            autoencoder.load_state_dict(torch.load('models/NT/mls_mandible_autoencoder-0.0035.pth'))
        elif category == "nb":
            autoencoder.load_state_dict(torch.load('models/NT/nasal_bone_autoencoder-0.0026.pth'))
        elif category == "ntaps":
            autoencoder.load_state_dict(torch.load('models/NT/ntaps_autoencoder-0.0029.pth'))
        elif category == "rbp":
            autoencoder.load_state_dict(torch.load('models/NT/rhombencephalon_autoencoder-0.0038.pth'))
        elif category == "thorax":
            autoencoder.load_state_dict(torch.load('models/NT/thorax_autoencoder-0.0053.pth'))
    elif view == "test":
        autoencoder.load_state_dict(torch.load('models/test.pth'))

    autoencoder.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(cropped, view, category):
    model_option(view, category)
    img = cropped.convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = resnet(img_tensor).squeeze().numpy()

    input_tensor = torch.tensor(feat).float().unsqueeze(0)
    with torch.no_grad():
        recon = autoencoder(input_tensor)

    error = nn.functional.mse_loss(recon, input_tensor).item()

    return error
