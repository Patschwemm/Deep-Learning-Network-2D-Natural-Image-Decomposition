import torch
import torch.nn as nn
from geometry_helper import distance_field_rectangle
from typing import Tuple


SMOOTH = 1e-6
# custom normalization distance to use for this dataset
# suppose we have 256x256 images, maximum distance is sqrt(256**2 + 256**2)
# after this maximum distance, the total distance is declining
shape = [256,256]
lin_shape_x = torch.linspace(start=1, end=shape[0], steps=shape[0])
lin_shape_y = torch.linspace(start=1, end=shape[1], steps=shape[1])
distances = torch.sqrt(lin_shape_x**2 + lin_shape_y**2)
norm = torch.sum(distances * shape[0]/2)

def coverage_loss(field_x: torch.Tensor, mask_y: torch.Tensor):
    """
    Input is the distance field of prediction x and mask of target y
    Coverage Loss between predited image x and target y. 
    The coverage loss penalizes if target y is not completely covered by prediction x.
    A sufficient condition is if the distance field of the assembled shape evaluates to zero 
    for all points on the surface.
    """
    #calculate the distances 
    coverage = field_x[mask_y]
    sum_coverage = coverage.sum() 

    return sum_coverage / norm

def cosistency_loss(field_x: torch.Tensor, mask_y: torch.Tensor):
    """
    Input is the distance field of prediction x and mask of target y
    Consistency Loss between predited image x and target y. 
    The Consistency Loss penalizes if the predicted shapes (Union of shapes)
    are not completely inside the target object O.
    A sufficient condition is if the distance field of the target object O evaluates to zero
    for all points on the surface.
    """
    #calculate the distances 
    print(mask_y)
    coverage = field_x[mask_y]
    sum_coverage = coverage.sum() 

    return sum_coverage / norm

x = torch.tensor([128., 128., 1., 1.], requires_grad=True)
y = torch.ones((256, 256))
mask_y = [y == 1]
field_pred = distance_field_rectangle(x,y)
print("coverage loss:", 
coverage_loss(field_pred, mask_y))


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


