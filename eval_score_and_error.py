import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ==== Chemins ====
pop_true_path = "data/raw/pop_patch1.tif"
pop_pred_path = "src/predicted_population.tif"

def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)

# ==== Lecture ====
true = read_raster(pop_true_path).astype(np.float32)
pred = read_raster(pop_pred_path).astype(np.float32)

# ==== Scores ====
mse = mean_squared_error(true.flatten(), pred.flatten())
mae = mean_absolute_error(true.flatten(), pred.flatten())
r2 = r2_score(true.flatten(), pred.flatten())

print("\n📊 Scores de performance :")
print(f"🔹 MSE (Mean Squared Error) : {mse:.2f}")
print(f"🔹 MAE (Mean Absolute Error) : {mae:.2f}")
print(f"🔹 R² Score                : {r2:.4f}")

# ==== Carte d’erreur absolue ====
error_map = np.abs(true - pred)

plt.figure(figsize=(10, 6))
plt.imshow(error_map, cmap='hot')
plt.colorbar(label="Erreur absolue")
plt.title("🗺️ Carte d’erreur entre population réelle et prédite")
plt.axis("off")
plt.tight_layout()
plt.show()
   
   # Export GeoTIFF erreur
export_path = "outputs/error_map.tif"
with rasterio.open(pop_true_path) as src:
    meta = src.meta
    meta.update(dtype=rasterio.float32)

    with rasterio.open(export_path, "w", **meta) as dst:
        dst.write(error_map.astype(np.float32), 1)

print(f"✅ Carte d’erreur exportée : {export_path}")
