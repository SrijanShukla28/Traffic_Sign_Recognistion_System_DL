import torch
from torchvision import transforms as T

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    """Converts a PIL Image or numpy.ndarray to tensor."""
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a given probability."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = T.functional.hflip(image)
            if "boxes" in target:
                bbox = target["boxes"]
                # Get image width
                if isinstance(image, torch.Tensor):
                    width = image.shape[2]
                else: # PIL Image
                    width = image.width
                
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = T.functional.hflip(target["masks"])
                
        return image, target

# This is the function your notebook calls
def get_transforms(train=False):
    """
    Returns the appropriate transforms for training or validation.
    """
    transforms = []
    # All datasets get converted to tensor
    transforms.append(ToTensor())
    if train:
        # Add horizontal flip for training data
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)