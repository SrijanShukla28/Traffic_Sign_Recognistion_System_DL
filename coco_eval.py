import json
import tempfile
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from collections import defaultdict
import copy

class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            iou_types = [iou_types]
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type) for iou_type in iou_types}
        self.img_ids = []
        self.eval_imgs = {iou_type: [] for iou_type in iou_types}

    def update(self, predictions):
        img_ids = list(predictions.keys())
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            
            # Use loadRes directly
            self.coco_eval[iou_type].cocoDt = self.coco_gt.loadRes(results)
            self.eval_imgs[iou_type] = list(predictions.keys())


    def synchronize_between_processes(self):
        # No distributed processing, so this is a no-op
        pass

    def accumulate(self):
        for iou_type in self.iou_types:
            print(f"Accumulating evaluation results for iou_type: {iou_type}...")
            # Set the image IDs to evaluate
            self.coco_eval[iou_type].params.imgIds = self.img_ids
            # Run evaluation
            self.coco_eval[iou_type].evaluate()
            self.coco_eval[iou_type].accumulate()

    def summarize(self):
        for iou_type in self.iou_types:
            print(f"IoU metric: {iou_type}")
            self.coco_eval[iou_type].summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError(f"Unknown iou_type: {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if "boxes" not in prediction or "scores" not in prediction or "labels" not in prediction:
                continue

            boxes = prediction["boxes"].tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            for box, score, label in zip(boxes, scores, labels):
                # convert [x1, y1, x2, y2] to [x1, y1, width, height]
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                
                result = {
                    "image_id": original_id,
                    "category_id": label,
                    "bbox": box,
                    "score": score,
                }
                coco_results.append(result)
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if "masks" not in prediction or "scores" not in prediction or "labels" not in prediction:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            masks = prediction["masks"]
            
            # rle conversion
            masks = masks.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
            masks = mask_util.encode(masks.numpy())

            for mask, score, label in zip(masks, scores, labels):
                result = {
                    "image_id": original_id,
                    "category_id": label,
                    "segmentation": mask,
                    "score": score,
                }
                coco_results.append(result)
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if "keypoints" not in prediction or "scores" not in prediction or "labels" not in prediction:
                continue

            keypoints = prediction["keypoints"].tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            for kps, score, label in zip(keypoints, scores, labels):
                result = {
                    "image_id": original_id,
                    "category_id": label,
                    "keypoints": kps,
                    "score": score,
                }
                coco_results.append(result)
        return coco_results