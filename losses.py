import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math
import kornia
from field_geometry import *


def coverage_loss(
    prim:torch.FloatTensor,
    mask:torch.BoolTensor,
    z: Optional[torch.BoolTensor]=None,
    mode: str=""
) -> torch.FloatTensor:
    """ Compute the coverage loss for rectangular primitives. 
        See Eq. (3) to (6) in Tulsiani et al. (2018)

        Args:
            prim (torch.FloatTensor): 
                primitive of shape:
                rect: (b, p, 2, 2) with r[:, :, 0, :] as x, r[:, :, 1, :] as y
                circ: (b, p, 1, 3) with c[:, :, 0, 1] as x, c[:, :, 0, 1] as y
                    and c[:, :, 0, 2] as radius
            mask (torch.BoolTensor):
                target mask to cover by primitives
            z (Optional[torch.BoolTensor]):
                valid mask for rectangles of shape (b, p).
    """
    # get batch-size and number of primitives
    b, p = prim.size(0), prim.size(1)
    z = z.bool() if z is not None else torch.ones((b, p), dtype=bool, device=prim.device)
    # compute all distance fields for all primitives in all batches
    # and apply valid mask afterwards
    if mode == "Rectangle":
        d = compute_rectangle_distance_field(prim.reshape(-1, 2, 2), torch.tensor([mask.size(1), mask.size(2)]))
    elif mode == "Circle":
        d = compute_circle_distance_field(prim.reshape(-1, 1, 3), torch.tensor([mask.size(1), mask.size(2)]))
    elif mode == "Triangle":
        pass
    # plt.imshow(d[0].detach().cpu(), cmap="gray")
    # plt.savefig("pictures/distancefield_outward.png")
    # plt.close()
    d = d.reshape(b, p, mask.size(1), mask.size(2))
    d = torch.masked_fill(d, ~z.reshape(b, p, 1, 1), float('inf'))
    d = d.min(dim=1).values
    # handle no primitive selected at all
    z_none = ~z.any(dim=1).reshape(b, 1, 1)
    d = torch.masked_fill(d, z_none, 0.0)
    # get the values of interest from distance field and compute average
    return ((d * mask).reshape(b, -1).mean(dim=1))

def coverage_loss_all(
    r:torch.FloatTensor,
    r_rot:torch.FloatTensor,
    c:torch.FloatTensor,
    mask:torch.BoolTensor,
    z: Optional[torch.BoolTensor]=None,
) -> torch.FloatTensor:
    """ Compute the coverage loss for rectangular primitives. 
        See Eq. (3) to (6) in Tulsiani et al. (2018)

        Args:
            r (torch.FloatTensor): 
                rectangle corners in range [0, 1]. Must be of shape (b, 2, 2)
                where the r[:, 0, :] specifies all x coordinates and r[:, 1, :]
                all y coordinates
            c (torch.FloatTensor): 
                rectangle corners in range [0, 1]. Must be of shape (b, 2, 2)
                where the c[:, 0, 0] specifies all x center and c[:, 0, 1]
                all y center
            mask (torch.BoolTensor):
                target mask to cover by primitives
            z (Optional[torch.BoolTensor]):
                valid mask for rectangles of shape (b, p).
    """
    # get batch size, primitive count and compute distance field for rectangles and circles
    # rectangle primitive count and distance field extraction
    if r != None:
        b = r.size(0)
        p_rect= r.size(1)
        d_rect = compute_rotated_rectangle_distance_field(r.reshape(-1, 2, 2), r_rot, torch.tensor([mask.size(1), mask.size(2)]))
        d_rect = d_rect.reshape(b, p_rect, mask.size(1), mask.size(2))
        device = r.device
    else:
        p_rect = 0
        d_rect = torch.tensor((), device=c.device)
    # circle primitive count and distance field extraction
    if c != None:
        b = c.size(0)
        p_circ= c.size(1)
        d_circ = compute_circle_distance_field(c.reshape(-1, 1, 3), torch.tensor([mask.size(1), mask.size(2)]))
        d_circ = d_circ.reshape(b, p_circ, mask.size(1), mask.size(2))
        device = c.device
    else:
        p_circ = 0
        d_circ = torch.tensor((), device=r.device)
    p = p_rect + p_circ
    # concatenate the tensor to get the union of several primitives
    d = torch.cat((d_rect, d_circ), dim = 1) 
    # compute z mask selection of the primitives
    z = z.bool() if z is not None else torch.ones((b, p), dtype=bool, device=device)

    d = torch.masked_fill(d, ~z.reshape(b, p, 1, 1), float('inf'))
    d = d.min(dim=1).values
    # handle no primitive selected at all
    z_none = ~z.any(dim=1).reshape(b, 1, 1)
    d = torch.masked_fill(d, z_none, 0.0)

    # get the values of interest from distance field and compute average
    return ((d * mask).reshape(b, -1).mean(dim=1))

def consistency_loss(
    prim:torch.FloatTensor,
    p_rot:torch.FloatTensor,
    mask:torch.BoolTensor,
    z:Optional[torch.BoolTensor]=None,
    mode: str=""
) -> torch.FloatTensor:
    """ Compute consistency loss for rectangles

        Args:
            prim (torch.FloatTensor): 
                primitive of shape:
                rect: (b, p, 2, 2) with r[:, :, 0, :] as x, r[:, :, 1, :] as y
                circ: (b, p, 1, 3) with c[:, :, 0, 0] as x, c[:, :, 0, 1] as y
                    and c[:, :, 0, 2] as radius
            mask (torch.BoolTensor):
                target mask to cover by rectangle primitives
            z (Optional[torch.BoolTensor]):
                valid mask for rectangles of shape (b, p).
    """
    # get batch-size and number of primitives
    b, p = prim.size(0), prim.size(1)
    z = z.bool() if z is not None else torch.ones((b, p), dtype=bool, device=prim.device)
    # compute inwards rectangle distance fields
    if mode == "Rectangle":
        d = compute_rotated_rectangle_distance_field(prim.reshape(-1, 2, 2), p_rot, torch.tensor([mask.size(1), mask.size(2)]), inwards=True)
    elif mode == "Circle":
        d = compute_circle_distance_field(prim.reshape(-1, 1, 3), torch.tensor([mask.size(1), mask.size(2)]), inwards=True)
    elif mode == "Triangle":
        pass
    plt.imshow(d[0].squeeze().detach().cpu(), cmap="gray")
    plt.savefig("pictures/distancefield_inward.png")
    plt.close()
    d = d.reshape(b, p, mask.size(1), mask.size(2)).permute(0, 2, 3, 1)
    # apply rectangle mask
    d = torch.masked_fill(d, ~z.reshape(b, 1, 1, p), 0.0)
    # compute consistency loss
    loss = (d * ~mask.bool().unsqueeze(-1)).sum(dim=-1)
    return loss.reshape(b, -1).mean(dim=1)


