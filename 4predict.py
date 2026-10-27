"""
Step 4 - Run inference on large GeoTIFF scenes and export predictions.

Tiles a large image with an 8192 px sliding window (10% overlap), then
sub-tiles each window to 512 px for Mask R-CNN inference, stitches the
predicted masks back into a scene-sized mask, extracts contours, and
converts them to geo-referenced polygons saved as a Shapefile.
"""

import cv2
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from tqdm import tqdm
from rasterio.transform import xy
from rasterio.windows import Window
import os

# Configure Detectron2 predictor
cfg = get_cfg()
cfg.merge_from_file(
    r"D:\Anaconda3\envs\tensorflow\Lib\site-packages\detectron2\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = r"D:\pyprojects\ml_algoritms\Mask R-CNN\output\model.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)

# Read the large image
image_path = r"D:\pyprojects\ml_algoritms\Mask R-CNN\predict\1.tif"
with rasterio.open(image_path) as dataset:
    H, W = dataset.shape  # Get the original image dimensions

# Large tile parameters
large_tile_size = 8192  # Size of each large tile
overlap_ratio = 0.1  # 10% overlap
large_stride = int(large_tile_size * (1 - overlap_ratio))  # Compute stride for large tiles

predictions = []
final_masks = []  # Store local masks

# Loop through large tiles (8192×8192)
for ly in tqdm(range(0, H, large_stride)):
    for lx in tqdm(range(0, W, large_stride)):
        lx_end = min(lx + large_tile_size, W)
        ly_end = min(ly + large_tile_size, H)
        window = Window(lx, ly, lx_end - lx, ly_end - ly)
        with rasterio.open(image_path) as dataset:
            img_array = dataset.read([1, 2, 3], window=window).astype(np.uint8)  # Read as uint8
            img_array = np.moveaxis(img_array, 0, -1)  # Convert from (C, H, W) to (H, W, C)
        large_tile = img_array
        local_mask = np.zeros((large_tile.shape[0], large_tile.shape[1]), dtype=np.uint8)  # Initialize local mask

        # Small tile parameters
        tile_size = 512  # Fine-grained tile size
        stride_ratio = 0.75
        stride = int(tile_size * stride_ratio)  # Compute stride for small tiles

        # Loop through small tiles
        for y in range(0, large_tile.shape[0], stride):
            for x in range(0, large_tile.shape[1], stride):
                x_end = min(x + tile_size, large_tile.shape[1])
                y_end = min(y + tile_size, large_tile.shape[0])
                tile = large_tile[y:y_end, x:x_end]
                if tile.shape[0] < 100 or tile.shape[1] < 100:
                    continue  # Skip very small tiles

                # Run Mask R-CNN prediction
                outputs = predictor(tile)
                instances = outputs["instances"].to("cpu")

                # Parse masks
                masks = instances.pred_masks.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()

                # Update local mask for each instance
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    mask = masks[i]
                    # Convert to local coordinates
                    local_mask[y:y_end, x:x_end] = np.maximum(local_mask[y:y_end, x:x_end], mask.astype(np.uint8) * 255)
                    predictions.append({
                        "bbox": [x1 + lx + x, y1 + ly + y, x2 + lx + x, y2 + ly + y],
                        "mask": local_mask,  # Store local mask
                        "score": scores[i],
                        "class": classes[i]
                    })

        final_masks.append((lx, ly, local_mask))  # Store local mask with its origin coordinates

# Merge all local masks into a global mask
final_mask = np.zeros((H, W), dtype=np.uint8)
for lx, ly, mask in final_masks:
    final_mask[ly:ly + mask.shape[0], lx:lx + mask.shape[1]] = np.maximum(
        final_mask[ly:ly + mask.shape[0], lx:lx + mask.shape[1]], mask
    )

# Contour detection
contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Store pixel coordinates (without converting to geocoordinates yet)
shp_save_path = r"D:\pyprojects\ml_algoritms\Mask R-CNN\try\85.shp"
polygons = []

with rasterio.open(image_path) as src:
    transform = src.transform  # Get affine transform
    crs = src.crs  # Get coordinate system

# Convert pixel contours to polygons in geographic coordinates
for contour in contours:
    if len(contour) >= 3:  # Skip very small areas
        contour = contour.reshape(-1, 2)
        geo_coords = [xy(transform, y, x) for x, y in contour]  # Convert to geocoordinates
        polygon = Polygon(geo_coords)
        polygons.append(polygon)

# Create Shapefile
gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
gdf.to_file(shp_save_path)

print(f"✅ Mask has been converted to Shapefile: {shp_save_path}")
