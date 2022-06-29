import torch
import torch.nn as nn
from geometry_helper import distance_field_rectangle
from geometry_helper import distance_field_batch
from typing import Tuple
import matplotlib.pyplot as plt


def all_loss_fn(x: torch.tensor,  mask_y:torch.tensor, field_y: torch.tensor, device: torch.device):

    # print("x shape for batching of bbox", x.shape)
    x_reshaped = torch.reshape(x, (x.shape[0], 2, 2))

    field_x = distance_field_batch(x_reshaped, field_y.shape, device=device).to(device=device)
    # print(field_x.shape)
    # plt.imshow(null_fields[0],cmap="gray")
    # plt.savefig("pictures/null_fields.png")

    # mask_x = field_x.masked_fill_(field_x == 0, 1)
    mask_x = (field_x == 0).to(device=device)
    # print(mask_x.shape)


    # print(mask_x.max())
    # plt.imshow(mask_x[0],cmap="gray")
    # plt.savefig("pictures/mask_x.png")
    # plt.close()

    # print(mask_x.shape)


    cov_loss = coverage_loss(field_x=field_x, mask_y=mask_y)
    consis_loss = consistency_loss(field_y=field_y, mask_x=mask_x)

    return cov_loss + consis_loss

def coverage_loss(field_x: torch.Tensor, mask_y: torch.Tensor):
    """
    Input is the distance field of prediction x and mask of target y
    Coverage Loss between predited image x and target y. 
    The coverage loss penalizes if target y is not completely covered by prediction x.
    A sufficient condition is if the distance field of the assembled shape evaluates to zero 
    for all points on the surface.
    """
    #calculate the distances 
    coverage = field_x * mask_y
    sum_coverage = coverage.mean() 

    return sum_coverage

def consistency_loss(field_y: torch.Tensor, mask_x: torch.Tensor):
    """
    Input is the distance field of prediction x and mask of target y
    Consistency Loss between predited image x and target y. 
    The Consistency Loss penalizes if the predicted shapes (Union of shapes)
    are not completely inside the target object O.
    A sufficient condition is if the distance field of the target object O evaluates to zero
    for all points on the surface.
    """
    #calculate the distances 
    coverage = field_y * mask_x
    sum_coverage = coverage.mean()

    return sum_coverage

