import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

print("Libraries loaded....")
cat_list = ["abdomen", "diencephalon", "head", "lateral_ventricle", "maxilla", "mds_mandible", "mls_mandible", "nasal_bone", "ntaps", "nuchal_translucency", "rhombencephalon", "thorax"]
dir = 'NT' #CRL

for cat in cat_list:
    normal_feats = np.load(f"features/{dir}/normal_{cat}_features.npy")
    print(f"{cat} features loaded")

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
    print("Architecture defined")
    autoencoder = AE()
    opt = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train/val split
    train, val = train_test_split(normal_feats, test_size=0.2)
    train_loader = DataLoader(torch.tensor(train).float(), batch_size=16, shuffle=True)
    val_loader = DataLoader(torch.tensor(val).float(), batch_size=16)

    # Training loss
    print("Training ongoing....")
    for epoch in range(30):
        autoencoder.train()
        loss_sum = 0
        for batch in train_loader:
            opt.zero_grad()
            out = autoencoder(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch+1} | Train Loss: {loss_sum/len(train_loader):.4f}")
    print("Training complete")
    torch.save(autoencoder.state_dict(), f'models/{dir}/{cat}_autoencoder-{loss_sum/len(train_loader):.4f}.pth')

