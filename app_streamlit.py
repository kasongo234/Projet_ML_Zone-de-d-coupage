import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import zipfile
import shutil
from rasterio.features import shapes
from shapely.geometry import shape
import os

# === Fonctions ===

def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs

def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return round(mse, 2), round(mae, 2), round(r2, 3)

def export_zd_to_shapefile(raster_path, output_folder="outputs/zd_shapefile"):
    data, transform, crs = load_raster(raster_path)
    os.makedirs(output_folder, exist_ok=True)
    mask = data > 0
    results = (
        {"geometry": shape(geom), "properties": {"cluster": int(val)}}
        for geom, val in shapes(data, mask=mask, transform=transform)
    )
    gdf = gpd.GeoDataFrame.from_features(results)
    gdf.set_crs(crs, inplace=True)
    shp_path = os.path.join(output_folder, "zd_advanced.shp")
    gdf.to_file(shp_path)
    return output_folder

def zip_folder(folder_path, zip_name):
    shutil.make_archive(zip_name, 'zip', folder_path)
    return zip_name + ".zip"

# === Configuration Streamlit ===
st.set_page_config(page_title="Estimation population", layout="wide")
st.title("🧠 Estimation de la population à partir d’images satellites")

# === Fichiers disponibles
paths = {
    "Population réelle": "data/raw/pop_patch1.tif",
    "Population prédite": "src/predicted_population.tif",
    "Carte d’erreur (diff absolue)": "outputs/error_map.tif",
    "Zones de Dénombrement (ZD)": "outputs/zd_kmeans.tif",
    "Zones de Dénombrement (ZD avancées)": "outputs/zd_advanced.tif"
}

option = st.sidebar.selectbox("🗺️ Choisissez la carte à afficher :", list(paths.keys()))
data, transform, crs = load_raster(paths[option])

# === Affichage principal
col1, col2 = st.columns([4, 2])
with col1:
    fig, ax = plt.subplots(figsize=(8, 6))

    if "ZD" in option:
        cmap = "tab20"
        masked = np.ma.masked_where(data < 0, data)
        im = ax.imshow(masked, cmap=cmap)
    else:
        cmap = "hot" if "erreur" in option else "viridis"
        im = ax.imshow(data, cmap=cmap)

    ax.set_title(option, loc="center", fontsize=14)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.6)
    st.pyplot(fig)

# === Statistiques
with col2:
    if option == "Population prédite":
        true_data, _, _ = load_raster(paths["Population réelle"])
        mse, mae, r2 = compute_metrics(true_data.flatten(), data.flatten())
        st.metric("MSE (Erreur quadratique)", mse)
        st.metric("MAE (Erreur absolue)", mae)
        st.metric("Score R²", r2)

        # Résumé automatique
        st.markdown("---")
        st.subheader("📊 Analyse automatique :")
        if r2 < 0:
            st.warning("❗ Le modèle n’apprend pas correctement. Envisagez d’augmenter les données ou d'approfondir le réseau.")
        elif r2 < 0.5:
            st.info("🟡 Le modèle apprend partiellement. Certaines zones sont mal estimées.")
        else:
            st.success("🟢 Le modèle prédit bien la répartition spatiale.")

    elif "ZD" in option:
        n_zones = len(np.unique(data[data >= 0]))
        st.metric("📂 Nombre de zones détectées", n_zones)

# === Export shapefile si ZD avancées
if option == "Zones de Dénombrement (ZD avancées)":
    if st.button("📤 Exporter les ZD avancées en Shapefile (.zip)"):
        folder = export_zd_to_shapefile(paths[option])
        zip_path = zip_folder(folder, "zd_shapefile")
        with open(zip_path, "rb") as f:
            st.download_button("📥 Télécharger le Shapefile", f, file_name="zd_avancees.zip")
