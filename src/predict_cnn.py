import torch
import torch.nn as nn
import rasterio
import numpy as np
from train_cnn import PopCNN, PopulationDataset  # On réutilise la classe modèle et dataset
import os

# 📂 Chemins relatifs robustes
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
model_path = os.path.abspath(os.path.join(base_path, "..", "pop_model_simul.pth"))

output_path = os.path.join(base_path, "predicted_population.tif")

# 🔄 Rechargement des données
image_paths = [
    "data/raw/ndvi_patch1.tif",
    "data/raw/urban_patch1.tif",
    "data/raw/viirs_patch1.tif"
]

label_path = "data/raw/pop_patch1.tif"  # Juste pour récupérer la géométrie

# Charger données
dataset = PopulationDataset(image_paths, label_path)
x, _ = dataset[0]
x = x.unsqueeze(0)  # (1, 3, H, W)

# Charger modèle
model = PopCNN()
model.load_state_dict(torch.load(model_path))
model.eval()

# Prédiction
with torch.no_grad():
    prediction = model(x).squeeze(0).squeeze(0).numpy()  # (H, W)

# Sauvegarde raster
with rasterio.open(label_path) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction.astype(rasterio.float32), 1)

print("✅ Carte prédite sauvegardée :", output_path)
