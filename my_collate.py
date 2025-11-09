"""
This file contains the custom collate_fn for the DataLoader.
"""

def collate_fn(batch):
    return tuple(zip(*batch))