import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os

import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
image_paths = [
    os.path.join(base_path, "data", "raw", "ndvi_patch1.tif"),
    os.path.join(base_path, "data", "raw", "urban_patch1.tif"),
    os.path.join(base_path, "data", "raw", "viirs_patch1.tif")
]
label_path = os.path.join(base_path, "data", "raw", "pop_patch1.tif")


# ==== Dataset personnalisé ====

class PopulationDataset(Dataset):
    def __init__(self, image_paths, label_path):
        self.image_paths = image_paths
        self.label_path = label_path
        self.images = [self.load_raster(p) for p in self.image_paths]
        self.labels = self.load_raster(self.label_path)

    def load_raster(self, path):
        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float32)
        return array

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = np.stack(self.images, axis=0)
        y = np.expand_dims(self.labels, axis=0)
        return torch.tensor(x), torch.tensor(y)

# ==== Modèle CNN ====
class PopCNN(nn.Module):
    def __init__(self):
        super(PopCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)
    
# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Modèle ====
model = PopCNN()
model = model.to(device)

# ==== Loss & Optimiseur ====
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== Chemins ====

image_paths = [
    "data/raw/ndvi_patch1.tif",
    "data/raw/urban_patch1.tif",
    "data/raw/viirs_patch1.tif"
]
label_path = "data/raw/pop_patch1.tif"

# ==== Entraînement ====

dataset = PopulationDataset(image_paths, label_path)
loader = DataLoader(dataset, batch_size=1)

model = PopCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    for x, y in loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

# ==== Sauvegarde ====
torch.save(model.state_dict(), "pop_model_simul.pth")
print("✅ Entraînement terminé et modèle sauvegardé.")
