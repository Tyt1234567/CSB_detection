"""
Step 5 - Evaluate predictions against ground truth.

Rasterizes the predicted and ground-truth Shapefiles onto a common grid and
computes pixel-level precision, recall, and F1 score.
"""

import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize

# Input vector files
shp_true = r"D:\pyprojects\ml_algoritms\Mask R-CNN\predict\caigangfang_val.shp"  # Ground Truth
shp_pred = r"D:\pyprojects\ml_algoritms\Mask R-CNN\output\95.shp"  # Prediction

# Set pixel resolution (in meters)
pixel_size = 0.1

# Read vector data
gdf_true = gpd.read_file(shp_true)
gdf_pred = gpd.read_file(shp_pred)

# Align the bounding box for rasterization
minx, miny, maxx, maxy = gdf_true.total_bounds
minx_p, miny_p, maxx_p, maxy_p = gdf_pred.total_bounds
minx, miny, maxx, maxy = min(minx, minx_p), min(miny, miny_p), max(maxx, maxx_p), max(maxy, maxy_p)

# Compute raster dimensions
width = int((maxx - minx) / pixel_size)
height = int((maxy - miny) / pixel_size)

# Rasterization function
def rasterize_shp(gdf, width, height, bounds):
    """
    Convert a GeoDataFrame to a raster array.
    """
    shapes = [(geom, 1) for geom in gdf.geometry]
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=rasterio.transform.from_bounds(*bounds, width, height),
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    return raster

# Rasterize Ground Truth and Prediction
raster_true = rasterize_shp(gdf_true, width, height, (minx, miny, maxx, maxy))
raster_pred = rasterize_shp(gdf_pred, width, height, (minx, miny, maxx, maxy))

# Compute True Positive, False Positive, False Negative
tp = np.sum((raster_true == 1) & (raster_pred == 1))  # Correct prediction
fp = np.sum((raster_true == 0) & (raster_pred == 1))  # False positive
fn = np.sum((raster_true == 1) & (raster_pred == 0))  # Missed detection

# Compute evaluation metrics
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Output results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
