"""
Step 1 - Clip training tiles from large GeoTIFF imagery.

Reads color-steel building polygons from a shapefile, converts their
coordinates to pixel space, crops an appropriately sized tile around each
polygon (512 / 1024 / 2048 px depending on object size), and writes out the
cropped PNG tiles together with a COCO-format annotation file.
"""

import json
import geopandas as gpd
import numpy as np
import rasterio
import os
from shapely.geometry import Polygon, box
from rasterio.windows import Window
import cv2

def transform_shp_to_pixel(shp_file, tif_file):
    with rasterio.open(tif_file) as src:
        transform = src.transform
        shp_data = gpd.read_file(shp_file)
        # Remove None values
        shp_data = shp_data.dropna(subset=['geometry'])
        # Ensure all geometries are Polygon or MultiPolygon
        shp_data = shp_data[shp_data.geometry.apply(
            lambda g: isinstance(g, (Polygon, type(None))) or g.geom_type in ["Polygon", "MultiPolygon"])]
        # Transform coordinates
        shp_data['pixel_geometry'] = shp_data.geometry.apply(lambda geom: transform_geometry_to_pixel(geom, transform))
        # Remove objects that failed transformation
        shp_data = shp_data.dropna(subset=['pixel_geometry'])
    return shp_data

def transform_geometry_to_pixel(geometry, transform):
    if isinstance(geometry, Polygon):
        new_coords = [(~transform * (x, y)) for x, y in geometry.exterior.coords]
        return Polygon(new_coords)
    return geometry


def process_image_and_annotations(tif_file, image_id, annotation_id, output_folder, shp_data, transform, annotations):
    os.makedirs(output_folder, exist_ok=True)

    image_info_list = []
    annotation_list = []

    with rasterio.open(tif_file) as src:
        for _, row in shp_data.iterrows():
            pixel_polygon = row['pixel_geometry']
            bbox = pixel_polygon.bounds  # (xmin, ymin, xmax, ymax)

            raw_width = bbox[2] - bbox[0]
            raw_height = bbox[3] - bbox[1]
            if max(raw_width, raw_height) < 200:
                # Compute expanded crop region
                expand_x = (512 - raw_width) / 2
                expand_y = (512 - raw_height) / 2
            elif 600 > max(raw_width, raw_height) >= 200:
                expand_x = (1024 - raw_width) / 2
                expand_y = (1024 - raw_height) / 2
            else:
                expand_x = (2048 - raw_width) / 2
                expand_y = (2048 - raw_height) / 2
            x_min = max(0, int(bbox[0] - expand_x))
            y_min = max(0, int(bbox[1] - expand_y))
            x_max = min(src.width, int(bbox[2] + expand_x))
            y_max = min(src.height, int(bbox[3] + expand_y))

            width, height = x_max - x_min, y_max - y_min

            # Crop and save image
            output_file, x_min, y_min, width, height = cut_tif(x_min, y_min, x_max, y_max, tif_file, image_id,
                                                               output_folder)
            if output_file is None:  # Skip if invalid
                continue

            # Save image information
            image_info = {"id": image_id, "file_name": output_file, "width": width, "height": height}
            image_info_list.append(image_info)

            # Compute object positions relative to the cropped image
            window_geom = box(x_min, y_min, x_max, y_max)
            intersecting_objects = shp_data[shp_data['pixel_geometry'].intersects(window_geom)]

            for _, obj in intersecting_objects.iterrows():
                poly = obj['pixel_geometry'].intersection(window_geom)
                if poly.is_empty:
                    continue

                bbox_new = [poly.bounds[0] - x_min, poly.bounds[1] - y_min,
                            poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]]
                segmentation = []

                if poly.geom_type == "Polygon":
                    segmentation.append(sum([list((x - x_min, y - y_min)) for x, y in
                                             zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])], []))
                elif poly.geom_type == "MultiPolygon":
                    for p in poly.geoms:
                        segmentation.append(sum([list((x - x_min, y - y_min)) for x, y in
                                                 zip(p.exterior.coords.xy[0], p.exterior.coords.xy[1])], []))

                annotation_list.append(create_coco_annotation(annotation_id, image_id, 1, bbox_new, segmentation))
                annotation_id += 1

            image_id += 1  # Increment image_id

    return image_info_list, annotation_list, annotation_id


def cut_tif(x_min, y_min, x_max, y_max, tif_file, image_id, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    with rasterio.open(tif_file) as src:
        width, height = x_max - x_min, y_max - y_min
        if width <= 0 or height <= 0:
            print(f"⚠️ Invalid crop region at {x_min}, {y_min}, skipping...")
            return None, None, None, None, None

        window = Window(x_min, y_min, width, height)
        cropped_image = np.moveaxis(src.read(window=window), 0, -1)
        # Check the shape of the cropped image
        if cropped_image is None or cropped_image.size == 0:
            print(f"⚠️ Empty cropped image at {x_min}, {y_min}")
            return None, None, None, None, None
        print(f"Cropped Image Shape: {cropped_image.shape}, Type: {cropped_image.dtype}")
        # Check if data is valid
        if np.isnan(cropped_image).any():
            print("⚠️ Cropped image contains NaN values!")
            return None, None, None, None, None
        if cropped_image.dtype != np.uint8:
            cropped_image = ((cropped_image - cropped_image.min()) / (
                        cropped_image.max() - cropped_image.min()) * 255).astype(np.uint8)

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        output_file = os.path.abspath(os.path.join(output_folder, f"tile_{image_id}.png"))
        # Attempt to save and check success
        success = cv2.imwrite(output_file, cropped_image)
        if success:
            print(f"✅ Saved: {output_file}")
        return output_file, x_min, y_min, width, height


def create_coco_annotation(annotation_id, image_id, category_id, bbox, segmentation):
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0
    }


if __name__ == '__main__':
    for i in range(1, 29):
        shapefile = fr'D:\image_data\hurong_railway\{i}\colorsteel.shp'
        tif_file = fr'D:\image_data\hurong_railway\{i}\{i}.tif'
        output_folder = rf"D:\pyprojects\ml_algoritms\Mask R-CNN\data\{i}"
        os.makedirs(output_folder, exist_ok=True)

        # Transform shapefile coordinates
        shp_data = transform_shp_to_pixel(shapefile, tif_file)
        image_id, annotation_id = 1, 1
        images, annotations = [], []

        # Process images and annotations
        image_info_list, annotation_list, annotation_id = process_image_and_annotations(
            tif_file, image_id, annotation_id, output_folder, shp_data, None, annotations
        )

        images.extend(image_info_list)
        annotations.extend(annotation_list)

        # Save COCO annotations
        coco_json = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "color_steel"}]}
        with open(os.path.join(output_folder, "coco_annotations.json"), 'w') as f:
            json.dump(coco_json, f, indent=4)

        print(f"COCO annotation file generated! ({output_folder}/coco_annotations.json)")
