import torch
from pycocotools.coco import COCO

def get_coco_api_from_dataset(dataset):
    """
    Helper function to get COCO api instance from a dataset.
    """
    # Check if dataset is a subset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
        
    # Check if dataset is the COCO class
    if isinstance(dataset, COCO):
        return dataset

    # Check if dataset has a 'coco' attribute
    if hasattr(dataset, 'coco'):
        return dataset.coco

    # If dataset is a list or tuple of datasets (e.g. ConcatDataset)
    if isinstance(dataset, (list, tuple)):
        if all(hasattr(d, 'coco') for d in dataset):
            # This is simplified: assumes all datasets share the same coco api
            return dataset[0].coco

    # Fallback: create a dummy COCO api
    # This is necessary if the dataset doesn't have a COCO api (e.g., custom dataset)
    # We create a simple one based on image_ids and categories
    
    # Try to get categories from the dataset
    categories = []
    if hasattr(dataset, 'get_categories'):
         categories = dataset.get_categories()
    elif hasattr(dataset, 'categories'):
         categories = dataset.categories
    else:
         # Create dummy categories if not available (1-indexed)
         print("Warning: Creating dummy COCO API. Category mapping might be incorrect.")
         # Let's try to infer from the dataset's targets
         all_labels = set()
         if hasattr(dataset, '__getitem__'):
             # This is slow, but a last resort
             # print("Inferring categories from dataset items... This might be slow.")
             for idx in range(min(len(dataset), 100)): # Check first 100
                 try:
                     _, target = dataset[idx]
                     if 'labels' in target:
                         all_labels.update(target['labels'].tolist())
                 except:
                     pass # __getitem__ might not return (img, target)
         
         if not all_labels:
             print("Could not infer categories. Evaluation might fail or be incorrect.")
             all_labels = set(range(1, 92)) # Default to 91 COCO classes
             
         categories = [{"id": int(i), "name": str(i), "supercategory": "none"} for i in sorted(list(all_labels))]


    coco_gt = COCO()
    coco_gt.dataset = {'images': [], 'annotations': [], 'categories': categories}
    
    image_ids = list(range(len(dataset))) # Use index as image_id
    coco_gt.dataset['images'] = [{'id': img_id} for img_id in image_ids]

    # Create dummy annotations
    ann_id = 1
    for idx, img_id in enumerate(image_ids):
        target = None
        try:
            _, target = dataset[idx]
        except:
            continue
            
        boxes = target.get('boxes', [])
        labels = target.get('labels', [])
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        for box, label in zip(boxes, labels):
            box_xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': label,
                'bbox': box_xywh,
                'area': box_xywh[2] * box_xywh[3],
                'iscrowd': 0
            }
            coco_gt.dataset['annotations'].append(ann)
            ann_id += 1
            
    coco_gt.createIndex()
    return coco_gt