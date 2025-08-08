import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import geopandas as gpd
from shapely.geometry import box

# Crée les dossiers nécessaires
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/shapefiles/communes_demo", exist_ok=True)

# Paramètres raster
width = height = 256
transform = from_origin(0, 256, 1, 1)  # origin_x, origin_y, pixel_width, pixel_height

def create_raster(path, seed):
    data = np.random.rand(height, width).astype("float32") * seed
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs='+proj=latlong',
        transform=transform
    ) as dst:
        dst.write(data, 1)

# Créer 4 rasters simulés
create_raster("data/raw/ndvi_patch1.tif", 1.0)
create_raster("data/raw/urban_patch1.tif", 100)
create_raster("data/raw/viirs_patch1.tif", 500)
create_raster("data/raw/pop_patch1.tif", 1000)

print("✅ Rasters simulés générés.")

# Créer un shapefile fictif
polygone = box(0, 0, 256, 256)
gdf = gpd.GeoDataFrame({'id': [1], 'nom': ['Commune_test']}, geometry=[polygone], crs='EPSG:4326')
gdf.to_file("data/shapefiles/communes_demo/communes_demo.shp")

print("✅ Shapefile simulé généré.")
