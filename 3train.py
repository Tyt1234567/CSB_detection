"""
Step 3 - Train a Mask R-CNN model (Detectron2) on the merged dataset.

Registers the train/validation COCO datasets produced by `2merge_json_tif.py`,
then fine-tunes a `mask_rcnn_R_50_FPN_3x` model to segment color-steel
buildings. Includes a custom trainer that logs proposal-level
positive/negative statistics and an optional RPN proposal visualization hook.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import cv2
import numpy as np
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, Instances
from torchvision.ops import nms, box_iou
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import polygons_to_bitmask
from detectron2.config import get_cfg
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

# Register COCO datasets
train_json = "D:/pyprojects/ml_algoritms/Mask R-CNN/train/coco_annotations.json"
train_images = "D:/pyprojects/ml_algoritms/Mask R-CNN/train"
val_json = "D:/pyprojects/ml_algoritms/Mask R-CNN/validation/coco_annotations.json"
val_images = "D:/pyprojects/ml_algoritms/Mask R-CNN/validation"

register_coco_instances("color_steel_train", {}, train_json, train_images)
register_coco_instances("color_steel_val", {}, val_json, val_images)

# Define positive and negative samples
def define_positive_negative_samples(nms_boxes, gt_boxes, gt_masks):
    max_iou_values, _ = box_iou(nms_boxes, gt_boxes).max(dim=1)
    mask_overlap_ratios = []

    for i, box in enumerate(nms_boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        box_area = (x2 - x1) * (y2 - y1)
        proposal_mask = torch.zeros_like(gt_masks[0])
        proposal_mask[y1:y2, x1:x2] = 1
        intersection = (proposal_mask.unsqueeze(0) & gt_masks).sum(dim=(1, 2)).float()
        mask_overlap_ratios.append(intersection.max().item() / (box_area + 1e-6))

    positive_indices = []
    negative_indices = []
    for i, box in enumerate(nms_boxes):
        iou, mask_overlap = max_iou_values[i].item(), mask_overlap_ratios[i]
        if iou > 0.7 or mask_overlap > 0.3:
            positive_indices.append(i)
        elif iou < 0.3 and mask_overlap < 0.1:
            negative_indices.append(i)

    return positive_indices, negative_indices


# RPN visualization hook
class RPNVisualizationHook(HookBase):
    def __init__(self, cfg, period=100):
        self.cfg = cfg
        self.period = period

    def after_step(self):
        iteration = self.trainer.iter
        if iteration % self.period == 0:
            storage = get_event_storage()
            batch = self.trainer._data_loader_iter
            data = next(batch)
            model = self.trainer.model
            images = model.preprocess_image(data)
            features = model.backbone(images.tensor)
            proposals, _ = model.proposal_generator(images, features, [x["instances"].to("cuda") for x in data])

            # Visualize the first image
            proposal_boxes = proposals[0].proposal_boxes
            image = data[0]["image"].cpu().numpy().transpose(1, 2, 0)
            visualizer = Visualizer(image, self.cfg)
            visualizer.draw_boxes(proposal_boxes.tensor.cpu())
            output = visualizer.get_output()

            output_file = os.path.join(self.cfg.OUTPUT_DIR, f"rpn_output_{iteration}.png")
            output.save(output_file)
            print(f"Saved RPN visualization to {output_file}")


class MyTrainer(DefaultTrainer):
    def run_step(self):
        self._trainer.iter += 1
        data = next(self._trainer._data_loader_iter)
        model = self._trainer.model
        optimizer = self._trainer.optimizer

        images = model.preprocess_image(data)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, [x["instances"].to("cuda") for x in data])

        all_anchors = proposals[0]
        proposals_tensor = all_anchors.proposal_boxes.tensor.cpu()
        gt_boxes = data[0]["instances"].gt_boxes.tensor.cpu()
        gt_polygons = data[0]["instances"].gt_masks.polygons
        height, width = data[0]["image"].shape[1:]
        gt_masks = torch.stack(
            [torch.tensor(polygons_to_bitmask(polygon, height, width)) for polygon in gt_polygons]).to(
            torch.uint8).cpu()

        # Only used to analyze proposal quality, no resampling
        positive_indices, negative_indices = define_positive_negative_samples(proposals_tensor, gt_boxes, gt_masks)

        # Compute loss
        loss_dict = model(data)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        get_event_storage().put_scalars(total_loss=losses.item(), **{k: v.item() for k, v in loss_dict.items()})

    def build_writers(self):
        return [
            CommonMetricPrinter(self.cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))
        ]


cfg = get_cfg()
cfg.merge_from_file(
    "D:/Anaconda3/envs/tensorflow/Lib/site-packages/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda"
cfg.DATASETS.TRAIN = ("color_steel_train",)
cfg.DATASETS.TEST = ("color_steel_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 50000
cfg.SOLVER.STEPS = [20000,40000]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RPN.NMS_THRESH = 0.6
cfg.MODEL.WEIGHTS = r"D:\pyprojects\ml_algoritms\Mask R-CNN\output\2.pth"
cfg.OUTPUT_DIR = "D:/pyprojects/ml_algoritms/Mask R-CNN/output"
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(cfg.MODEL)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
print(f"Model saved at {cfg.MODEL.WEIGHTS}")
