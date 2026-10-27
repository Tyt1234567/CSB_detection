# CSB_detection

Detection of hazardous **color-steel buildings** (CSB) in UAV / aerial imagery using **Mask R-CNN** (Detectron2). The project provides an end-to-end geospatial pipeline: turning shapefile-labeled GeoTIFF imagery into a COCO training set, training an instance-segmentation model, running inference on large orthomosaics, and evaluating the results against ground truth — all while keeping outputs in georeferenced vector formats (Shapefile) for direct use in GIS software.

Color-steel buildings (temporary structures with corrugated color-steel roofing) are frequently flagged as fire/safety hazards in urban areas. This pipeline automates their identification over large areas of UAV or satellite imagery to support risk assessment and urban safety analysis.

## Pipeline

The scripts are numbered in the order they should be run:

| Step | Script | Purpose |
|---|---|---|
| 1 | `1clip.py` | Reads building polygons from a shapefile, projects them into pixel space on the matching GeoTIFF, and crops an adaptively-sized tile (512 / 1024 / 2048 px) around each one. Produces PNG tiles + a per-region `coco_annotations.json`. |
| 2 | `2merge_json_tif.py` | Merges the per-region tiles/annotations from step 1 into a single dataset, renumbering image and annotation ids so they are globally unique. |
| 3 | `3train.py` | Registers the merged COCO dataset with Detectron2 and fine-tunes a `mask_rcnn_R_50_FPN_3x` model to segment color-steel buildings. |
| 4 | `4predict.py` | Runs the trained model over a full GeoTIFF scene using a two-level sliding window (8192 px tiles, sub-tiled to 512 px for inference), stitches the masks back together, and exports detections as a georeferenced Shapefile. |
| 5 | `5evaluate.py` | Rasterizes predicted vs. ground-truth Shapefiles on a common grid and reports pixel-level precision, recall, and F1 score. |

## Requirements

- Python 3.8+
- CUDA-capable GPU (training and inference use `cfg.MODEL.DEVICE = "cuda"`)

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

[Detectron2](https://github.com/facebookresearch/detectron2) must be installed separately following the [official instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) for your torch/CUDA version — it is not included in `requirements.txt`.

## Usage

Each script currently points at hardcoded local paths (e.g. `D:\pyprojects\ml_algoritms\Mask R-CNN\...`) at the top of the file or in its `if __name__ == "__main__":` block. Before running a step, open the script and update these paths to match your own data layout:

```bash
python 1clip.py            # build per-region tiles + annotations from shapefiles/GeoTIFFs
python 2merge_json_tif.py  # merge into a single COCO dataset
python 3train.py           # train Mask R-CNN
python 4predict.py         # run inference on a new GeoTIFF scene
python 5evaluate.py        # score predictions against ground truth
```

## Data format

- **Input labels**: Shapefile (`.shp`) polygons marking color-steel buildings, aligned with a GeoTIFF of the same area.
- **Training data**: COCO-format instance segmentation (`coco_annotations.json` + image tiles).
- **Predictions**: Shapefile polygons in the source image's CRS, ready to load alongside the original imagery in any GIS tool.
