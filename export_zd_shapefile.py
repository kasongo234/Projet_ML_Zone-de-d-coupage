import rasterio
from rasterio.features import shapes
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
import os

# Charger le raster ZD
raster_path = "outputs/zd_kmeans.tif"
shapefile_path = "outputs/zd_kmeans.shp"

with rasterio.open(raster_path) as src:
    image = src.read(1)
    mask = image != src.nodata

    results = (
        {'properties': {'zone': int(v)}, 'geometry': s}
        for s, v in shapes(image, mask=mask, transform=src.transform)
    )

    geoms = list(results)

# Convertir en GeoDataFrame
gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

# Sauvegarder en Shapefile
gdf.to_file(shapefile_path)

print(f"✅ Shapefile généré : {shapefile_path}")
