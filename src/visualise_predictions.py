import rasterio
import matplotlib.pyplot as plt
import os

# Chemins
pop_true = "data/raw/pop_patch1.tif"
pop_pred = "src/predicted_population.tif"

def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)

# Lecture des données
true_data = read_raster(pop_true)
pred_data = read_raster(pop_pred)

# Affichage
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(true_data, cmap='viridis')
axs[0].set_title("Population réelle (INSEE simulée)")
axs[0].axis("off")

axs[1].imshow(pred_data, cmap='magma')
axs[1].set_title("Population prédite (par le modèle)")
axs[1].axis("off")

plt.tight_layout()
plt.show()
