import torch
import torch.nn as nn
from geometry_helper import distance_field_rectangle
from geometry_helper import distance_field_batch
from typing import Tuple, Dict
import matplotlib.pyplot as plt


def all_loss_fn(x: torch.tensor, 
    prim_dict: Dict,  
    mask_y:torch.tensor, 
    field_y: torch.tensor,  
    device: torch.device, 
    consis_weight: float=1, 
    x_mask_padding: int=1):
    """
    Consis weight is used to adjust to magnitude of coverage loss. 
    Mask padding increases importance of the local environment around edges to be included in the loss. (higher values mean more padding)
    """

    rect_count = prim_dict["Rectangles"]
    tri_count = prim_dict["Triangles"]
    circles_count = prim_dict["Circles"]

    prim_count = rect_count + tri_count + circles_count

    x_reshaped = torch.reshape(x, (prim_count, x.shape[1], 2, 2))

    union_field = torch.ones_like(field_y) * float("inf")
    for i in range(prim_dict["Rectangles"]):
        field_x = distance_field_batch(x_reshaped[i], field_y.shape, field_y.size, device=device,).to(device=device)
        print(field_y.size)
        # plt.imshow(field_x[0].detach().cpu(),cmap="gray")
        # plt.savefig(f"pictures/field_x{i}.png")
        union_field = torch.minimum(field_x, union_field)
    
    # mask factor as penalty for very slim but long rectangles
    mask_x = (union_field * (-x_mask_padding))
    mask_x = torch.exp(mask_x)


    # print("field_y shape:", field_y.shape)
    # print("mask_y shape:", mask_y.shape)
    # print("field_x shape:", field_x.shape)
    # print("mask_x shape:", mask_x.shape)

    cov_loss = coverage_loss(field_x=union_field, mask_y=mask_y)
    consis_loss = consistency_loss(field_y=field_y, mask_x=mask_x)


    # plt.scatter(x=x[0, 0].detach().cpu(), y=x[0, 1].detach().cpu(), c='r', s=3)
    # plt.scatter(x=x[0, 2].detach().cpu(), y=x[0, 3].detach().cpu(), c='r', s=3)
    # plt.imshow(field_x[0].detach().cpu(),cmap="gray")
    # plt.savefig("pictures/field_x.png")
    # plt.imshow(mask_x[0].detach().cpu(),cmap="gray")
    # plt.savefig("pictures/mask_x.png")
    # plt.imshow(field_y[0],cmap="gray")
    # plt.savefig("pictures/field_y.png")
    # plt.imshow(mask_y[0],cmap="gray")
    # plt.savefig("pictures/mask_y.png")

    negative_penalty = -1 * torch.minimum(x, torch.zeros_like(x)).sum()
    # print(-negative_penalty * beta)
    # print(cov_loss + consis_weight * consis_loss + (-negative_penalty * beta))


    return  torch.log(1e-4 + cov_loss) + consis_weight * consis_loss + negative_penalty, [torch.log(1 + cov_loss) * 10, consis_weight * consis_loss, negative_penalty]

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
    consistency = field_y * mask_x
    sum_consistency = consistency.mean()

    return sum_consistency

