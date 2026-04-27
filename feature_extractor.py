import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

print("Libraries loaded...")

#config
cat = 'NT' #CRL
feature = 'nasal_bone'
img_paths = os.listdir(f"data/{cat}/{feature}_annotations")

#load pretrained Resnet-50 model and remove the classifier

weights = ResNet50_Weights.DEFAULT  
resnet = resnet50(weights=weights)
resnet.fc = nn.Identity()
resnet.eval()

#transform for Resnet input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        return resnet(x).squeeze().numpy()

#extract features

normal_feats = []
normal_files = os.listdir(f"data/{cat}/{feature}_annotations")
print("Feature extraction ongoing..")
for f in normal_files:
    path = os.path.join(f"data/{cat}/{feature}_annotations", f)
    normal_feats.append(extract_features(path))

normal_feats = np.array(normal_feats)
print("Features extracted")
np.save(f"features/{cat}/normal_{feature}_features.npy", normal_feats)
