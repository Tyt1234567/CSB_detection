"""
Step 2 - Merge per-region tiles and COCO annotations into one dataset.

Walks the output folders produced by `1clip.py`, renumbers image and
annotation ids so they are globally unique, copies/renames the tile images
into a single folder, and writes one combined `coco_annotations.json` ready
for training.
"""

import os
import shutil
import json
import re

def search_folders(path):
    folders = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            folders.append(os.path.join(root, dir_name))
    return folders

def search_tif(path):
    tif_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                tif_files.append(os.path.join(root, file))

    def extract_number(file_path):
        # Use regex to extract the number part from the filename
        match = re.search(r'tile_(\d+)\.png$', file_path)
        if match:
            return int(match.group(1))  # Convert the number part to integer
        return 0  # Return 0 if no match is found

    tif_files = sorted(tif_files, key=extract_number)
    return tif_files

def search_json(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                return data

# Find the relationship between tif filenames and tif.id, annotations.id, annotations.image_id
def parse_json(data):
    annotations = data["annotations"]
    relationships = []
    # Store (annotations.id, image_id) sequentially
    for annotation in annotations:
        relationships.append((annotation['id'], annotation['image_id']))
    return relationships

def copy_tif_with_new_name(src_file, dst_folder, new_name):
    # Ensure the destination folder exists; create if not
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    # Construct the target file path with the new filename
    dst_file = os.path.join(dst_folder, new_name)
    # Copy the file and rename it
    shutil.copy(src_file, dst_file)
    print(f"File copied and renamed to: {dst_file}")


if __name__ == "__main__":
    image_id = 1
    object_id = 1
    new_json = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "color_steel"
            }
        ]
    }

    folders = search_folders(r"D:\pyprojects\ml_algoritms\Mask R-CNN\new_test")

    for folder in folders:
        tifs = search_tif(folder)
        json_data = search_json(folder)
        box_id_image_id_relation = parse_json(json_data)
        old_new_image_id_relation = []

        # Process images field
        for idx, image in enumerate(json_data["images"]):
            image["id"] = image_id
            image["file_name"] = f"tile_{image_id}.png"

            copy_tif_with_new_name(tifs[idx], r"D:\pyprojects\ml_algoritms\Mask R-CNN\new_test",
                                   f"tile_{image_id}.png")
            new_json["images"].append(image)

            old_new_image_id_relation.append((idx + 1, image_id))
            image_id += 1

        # Process annotations field
        for idx, annotation in enumerate(json_data["annotations"]):
            box_id = annotation['id']
            old_image_id = box_id_image_id_relation[box_id - 1][1]
            new_image_id = old_new_image_id_relation[old_image_id - 1][1]
            annotation['image_id'] = new_image_id
            annotation['id'] = object_id

            new_json["annotations"].append(annotation)
            object_id += 1

    # Save the updated COCO JSON file
    with open(r"D:\pyprojects\ml_algoritms\Mask R-CNN\new_test\coco_annotations.json", 'w') as f:
        json.dump(new_json, f, indent=4)



