import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self, root, transform=None, dic=None, exts=(".jpg",".jpeg",".png",".ppm",".bmp",".tif",".tiff")):
        self.root = root
        self.transform = transform
        self.exts = exts

        # collect images recursively
        self.files = []
        for dp, dn, filenames in os.walk(self.root):
            for f in filenames:
                if f.lower().endswith(self.exts):
                    self.files.append(os.path.join(dp, f))
        self.files = sorted(self.files)
        print(f"myDataset: found {len(self.files)} images under {self.root}")

        # annotation dict keyed by basename (optional)
        self.dic = dic if dic is not None else {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Return (image, target) where:
         - image is a PIL image or transformed tensor
         - target is a dict with keys boxes, labels, image_id, area, iscrowd
        This version includes a BULLETPROOF data cleaning step to prevent 'nan' errors.
        """
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Get the actual width and height of the image
        img_width, img_height = img.size

        # prepare target from annotation dict
        basename = os.path.basename(img_path)
        objects = self.dic.get(basename, [])

        clean_boxes = []
        clean_labels = []

        for obj in objects:
            # --- !!! THIS IS THE BULLETPROOF 'nan' FIX !!! ---
            
            # 1. Get raw coordinates
            x1, y1, x2, y2 = obj['bbox']
            
            # 2. Fix "inverted" boxes by sorting
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)

            # 3. "Clip" the box to be 100% inside the image borders
            #    This is the most common cause of 'nan' errors
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            # 4. Check for "zero-area" boxes AFTER clipping
            if x_max <= x_min or y_max <= y_min:
                continue # Skip this bad annotation
            
            # If we get here, the box is 100% clean and safe
            clean_boxes.append([x_min, y_min, x_max, y_max])
            clean_labels.append(obj['label'])
            
            # --- !!! END OF FIX !!! ---

        if not clean_boxes:
            # If all boxes were bad, return an empty target
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(clean_boxes, dtype=torch.float32)
            labels = torch.tensor(clean_labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Apply transforms
        if self.transform:
            try:
                img, target = self.transform(img, target)
            except TypeError:
                img = self.transform(img)

        return img, target