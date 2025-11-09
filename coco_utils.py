import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

from collections import defaultdict
import copy
import numpy as np

# This is the new, corrected function
def convert_to_coco_api(ds):
    coco_ds = COCO()
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    
    # --- THIS IS THE FIX ---
    # We iterate through the dataset 'ds' (which is the Subset)
    # The 'img_idx' here will be 0 to 99 (for the 100 images in the test set)
    print(f"Converting {len(ds)} images from the test set to COCO format...")
    
    for img_idx in range(len(ds)):
        try:
            # Get the image and its targets from the dataset (this calls the Subset's __getitem__)
            img, targets = ds[img_idx]
            
            # The image_id is the *original* index, which is what we want
            image_id = targets["image_id"].item()
            
            # Create the image info dictionary
            img_dict = {}
            img_dict["id"] = image_id
            img_dict["height"] = img.shape[-2]
            img_dict["width"] = img.shape[-1]
            dataset["images"].append(img_dict)
            
            # Get annotation data
            bboxes = targets["boxes"]
            if bboxes.shape[0] > 0:
                # Convert [x1, y1, x2, y2] to [x1, y1, width, height]
                bboxes[:, 2:] -= bboxes[:, :2]
                bboxes = bboxes.tolist()
                labels = targets["labels"].tolist()
                areas = targets["area"].tolist()
                iscrowd = targets["iscrowd"].tolist()

                # Create annotation dictionaries for this image
                for i in range(len(bboxes)):
                    ann = {}
                    ann["image_id"] = image_id
                    ann["bbox"] = bboxes[i]
                    ann["category_id"] = labels[i]
                    categories.add(labels[i])
                    ann["area"] = areas[i]
                    ann["iscrowd"] = iscrowd[i]
                    # Give a unique ID to each annotation
                    ann["id"] = len(dataset["annotations"]) + 1
                    dataset["annotations"].append(ann)
        
        except Exception as e:
            # This might happen if an image in the test set has no annotations
            # print(f"Error or no annotations for image {img_idx} (Image ID: {image_id}): {e}")
            continue
            
    # Create the categories list
    dataset["categories"] = [{"id": i, "name": str(i), "supercategory": "none"} for i in sorted(categories)]
    
    # Load our fully-formed 'dataset' dict into the coco_ds object
    if dataset["annotations"]:
        coco_ds.dataset = dataset
        coco_ds.createIndex()
    else:
        print("--- WARNING: No annotations found in the test set! ---")

    print("Conversion to COCO format complete.")
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        # We NO LONGER want to unwrap the Subset
        # if isinstance(dataset, torch.utils.data.Subset):
        #     dataset = dataset.dataset
    
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
        
    # 'dataset' is our Subset. We pass it directly.
    return convert_to_coco_api(dataset)