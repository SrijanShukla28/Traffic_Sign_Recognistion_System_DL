"""
This is the fixed engine.py file.
It includes:
1. The main train_one_epoch function.
2. The Gradient Clipping fix to prevent 'nan' errors.
3. The 'evaluate' function is commented out to prevent all 'ghost file' errors.
"""

import math
import sys
import torch

import utils  # This imports the correct utils.py with MetricLogger

# We comment out the evaluation imports to prevent any 'ghost file' errors
# from coco_eval import CocoEvaluator
# from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            
            # --- !!! THIS IS THE 'nan' FIX: GRADIENT CLIPPING !!! ---
            # Unscale the gradients back for clipping
            scaler.unscale_(optimizer)
            # Clip the gradients to a max_norm of 1.0 (a safe value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # --- !!! END OF FIX !!! ---
            
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            
            # --- !!! THIS IS THE 'nan' FIX: GRADIENT CLIPPING !!! ---
            # Clip the gradients to a max_norm of 1.0 (a safe value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # --- !!! END OF FIX !!! ---
            
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


# --- We are skipping evaluation, so this function is simplified ---
@torch.no_grad()
def evaluate(model, data_loader, device):
    # This function is not being called by your Training Cell,
    # but we keep it here to prevent any import errors.
    print("--- Evaluation function is (safely) skipped. ---")
    pass