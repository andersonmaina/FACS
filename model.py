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
            autoencoder.load_state_dict(torch.load('models/abdomen_autoencoder-0.0058.pth'))
        elif category == "body":
            autoencoder.load_state_dict(torch.load('models/body_autoencoder-0.0060.pth'))
        elif category == "diencephalon":
            autoencoder.load_state_dict(torch.load('models/diencephalon_autoencoder-0.0050.pth'))
        elif category == "gsac":
            autoencoder.load_state_dict(torch.load('models/gestation_sac_autoencoder-0.0044.pth'))
        elif category == "head":
            autoencoder.load_state_dict(torch.load('models/head_autoencoder-0.0077.pth'))
        elif category == "lv":
            autoencoder.load_state_dict(torch.load('models/lateral_ventricle_autoencoder-0.0045.pth'))
        elif category == "mx":
            autoencoder.load_state_dict(torch.load('models/maxilla_autoencoder-0.0054.pth'))
        elif category == "mds":
            autoencoder.load_state_dict(torch.load('models/mds_mandible_autoencoder-0.0039.pth'))
        elif category == "mls":
            autoencoder.load_state_dict(torch.load('models/mls_mandible_autoencoder-0.0047.pth'))
        elif category == "nb":
            autoencoder.load_state_dict(torch.load('models/nasal_bone_autoencoder-0.0026.pth'))
        elif category == "ntaps":
            autoencoder.load_state_dict(torch.load('models/ntaps_autoencoder-0.0032.pth'))
        elif category == "rbp":
            autoencoder.load_state_dict(torch.load('models/rhombencephalon_autoencoder-0.0044.pth'))
        elif category == "thorax":
            autoencoder.load_state_dict(torch.load('models/thorax_autoencoder-0.0058.pth'))
    elif view == "nt":
        print("n/a")
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
