import torch
import torch.nn as nn
from geometry_helper import tensor_to_rectangle, intersection_area, union_area
from typing import Tuple


SMOOTH = 1e-6

def naive_IoU_rectangle(x,y):
    rect = tensor_to_rectangle(x, y)
    intersection = intersection_area(rect, y, output_tensor=False)
    union = union_area(rect, y, output_tensor=False)

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

def iou_pytorch(pred: torch.Tensor, target: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    pred = pred.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (pred & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred | target).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


