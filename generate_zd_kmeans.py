import rasterio
import numpy as np
from sklearn.cluster import DBSCAN

# === Charger l’image prédite ===
with rasterio.open("src/predicted_population.tif") as src:
    pop = src.read(1)
    transform = src.transform
    crs = src.crs
    profile = src.profile  # <- Important de récupérer ici

# === Extraction des pixels valides
rows, cols = np.where(pop > 0)
values = pop[rows, cols]
coords = np.array([[r, c] for r, c in zip(rows, cols)])

# === Clustering spatial avec DBSCAN
X = np.hstack([coords, values.reshape(-1, 1)])  # [row, col, pop]
clustering = DBSCAN(eps=3, min_samples=20).fit(X)

# === Création de la carte de labels
labels = -1 * np.ones_like(pop, dtype=np.int32)
for i, (r, c) in enumerate(zip(rows, cols)):
    labels[r, c] = clustering.labels_[i]

# === Sauvegarde en GeoTIFF
profile.update(dtype=rasterio.int32, count=1)
with rasterio.open("outputs/zd_advanced.tif", "w", **profile) as dst:
    dst.write(labels, 1)

print("✅ ZD avancées sauvegardées dans outputs/zd_advanced.tif")
