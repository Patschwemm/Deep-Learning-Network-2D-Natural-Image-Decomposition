import torch
from typing import Tuple, Dict, Optional
import math
import kornia
from matplotlib import pyplot as plt


def compute_rectangle_distance_field(
    r:torch.FloatTensor, 
    size:Tuple[int, int],
    inwards:bool=False
) -> torch.FloatTensor:
    """ Compute the distance field for a given rectangle.
        This function is backpropagatable.

        Args:
            r (torch.FloatTensor): 
                rectangle corners in range [0, 1]. Must be of shape (b, 2, 2)
                where the r[:, 0, :] specifies all x coordinates and r[:, 1, :]
                all y coordinates
            size (Tuple[int, int]): size of the generated distance field
    """
    # make sure the coordinates are sorted
    r = r.sort(dim=-1).values
    # scale rectangle coordinates from [0,1] to size
    scale = size.reshape(-1, 1).to(r.device)
    r = r * scale
    # build index mesh
    X = torch.arange(size[0], device=r.device)
    Y = torch.arange(size[1], device=r.device)
    # X = torch.linspace(0, 1, steps = size[0], device=r.device) 
    # Y = torch.linspace(0, 1, steps = size[1], device=r.device) 
    X, Y = X.reshape(1, -1, 1, 1), Y.reshape(1, 1, -1, 1)
    # compute L1 distance to each border in each dimension
    dX, dY = X - r[:, None, None, 0, :], Y - r[:, None, None, 1, :]
    dX[..., 0] *= -1
    dY[..., 0] *= -1
    # only keep distance to closet border
    # closest border has positive distance and the other
    # border has negative distance to query point
    dX = dX.max(dim=-1).values
    dY = dY.max(dim=-1).values
    # set interior to zero
    dX[(dX > 0) if inwards else (dX < 0)] = 0.0
    dY[(dY > 0) if inwards else (dY < 0)] = 0.0

    # compute distance
    # for inwards the distance is the minimum l1 distance, i.e. minimal distance to any border
    d_in = -torch.max(dX, dY)
    # for outwards the distance is the euclidean distance to the closest point on the rectangle
    d_out = (dX**2 + dY**2 + 1e-5).sqrt() - torch.sqrt(torch.tensor(1e-5))

    return d_in if inwards else d_out

def compute_rotated_rectangle_distance_field(
    r:torch.FloatTensor, 
    r_rot:torch.FloatTensor,
    size:Tuple[int, int],
    inwards:bool=False
) -> torch.FloatTensor:
    """ Compute the distance field for a given rectangle.
        This function is backpropagatable.

        Args:
            r (torch.FloatTensor): 
                rectangle corners in range [0, 1]. Must be of shape (b, 2, 2)
                where the r[:, 0, :] specifies all x coordinates and r[:, 1, :]
                all y coordinates
            size (Tuple[int, int]): size of the generated distance field
    """
    # expand image for the rotation to not get out of image bounds 
    image_size_rot = torch.tensor([int(math.sqrt(size[0]**2 + size[1]**2))] * 2, device=r.device)
    # if the size is not even, cropping can not be done properly, so check for even size
    image_size_rot += (1 if image_size_rot[0] % 2 == 1 else 0)
    # excess_pixels are calculated for bath axes
    excess_pixel = torch.div(image_size_rot - size[0], 2, rounding_mode='trunc').to(r.device)
    # adjust r to be in the same position as the in the original image size
    r_scaled = ((r * size[0]) + ((image_size_rot - size[0]) / 2)) / image_size_rot
    # center the rectangle coords and get the difference to the center to re-center afterwards
    r_centered, center_difference = center_rectangle(r_scaled, image_size_rot)

    # compute distance field of centered rectangle with scaled image
    d = compute_rectangle_distance_field(r_centered, image_size_rot, inwards)
    # rotate the distance field around the center
    d_rot = rotate_centered_field(d, r_rot, image_size_rot)
    # translate the distance field with re-center values to have original position
    d_translated = translate_field(d_rot.squeeze(1), center_difference, image_size_rot)
    # crop the translated rectangle to have the original pixel size
    d_crop = d_translated[:, :, excess_pixel[0]:-excess_pixel[0], excess_pixel[1]:-excess_pixel[1]]

    return d_crop


def compute_circle_distance_field(
    c:torch.FloatTensor, 
    size:Tuple[int, int],
    inwards:bool=False,
    plot: bool=False
) -> torch.FloatTensor:
    """ Compute the distance field for a given circle.
        This function is backpropagatable.
        Args:
            c (torch.FloatTensor): 
                rectangle corners in range [0, 1]. Must be of shape (b, 2, 2)
                where the c[:, 0, 0] specifies all x center and c[:, 0, 1]
                all y center
            size (Tuple[int, int]): size of the generated distance field
    """
    # make sure the coordinates are sorted
    # scale rectangle coordinates from [0,1] to size
    scale = size[-1].to(device=c.device)
    c = c * scale
    # build index mesh
    X = torch.arange(size[0], device=c.device)
    Y = torch.arange(size[1], device=c.device)
    # create grids for the distance function in L2 distance
    grid_x, grid_y = torch.meshgrid(X, Y, indexing='ij')
    # reshape for batch broadcast
    grid_x = grid_x.reshape(1, size[0], size[1], 1)
    grid_y = grid_y.reshape(1, size[0], size[1], 1)

    # compute L2 distance 
    d = torch.sqrt(
        (grid_x - c[:, None, None, None, 0, 0])**2 
        + (grid_y - c[:, None, None, None, 0, 1])**2 
        + 1e-5
        )
    # subtract radius of circle, 
    # negative values means inside radius, positive values mean outside
    # squeeze last dim for shape (b, :, :) instead of (b, :, :, 1)
    d = (d - c[:, 0, 2, None, None, None]).squeeze(-1)

    # inward: take negative values, outward: take positive values
    d[(d > 0) if inwards else (d < 0)] = 0.0


    # negative distances for inward so mult with -1
    return -d if inwards else d



def get_rectangle_mask(r:torch.FloatTensor, size:Tuple[int, int]) -> torch.BoolTensor:
    """ Get the mask filled by a rectangle
    
        Args:
            r (torch.FloatTensor): 
                rectangle corners in range [0, 1]. Must be of shape (b, 2, 2)
                where the r[:, 0, :] specifies all x coordinates and r[:, 1, :]
                all y coordinates
            size (Tuple[int, int]): size of the generated masks
    """
    # make sure the coordinates are sorted
    r = r.sort(dim=-1).values
    r = r.clamp(0, 1)
    # allocate memory for masks
    b = r.size(0)
    mask = torch.zeros((b, *size), dtype=bool)
    # scale rectangle coordinates from [0,1] to size
    scale = torch.tensor(size).reshape(-1, 1)
    r = (r * scale).round().long()
    # can't seem to do this without loops
    for i in range(b):
        # read rectangle corners
        x1, x2 = r[i, 0, :]
        y1, y2 = r[i, 1, :]
        # draw rectangle to mask
        mask[i, x1:x2, y1:y2] = True
    # return mask
    return mask

def center_rectangle(r, img_size):

    # get the centers of r and reshape to subtract according to batch size
    centers_difference = r.mean(dim=-1).reshape(1, -1).repeat(2, 1)
    centers_difference = centers_difference.mT
    centers_difference = centers_difference.reshape(r.shape) - 0.5

    # center the rectangle
    # to uncenter add centers_difference to r_centers
    r_centered = (r - centers_difference)
    return r_centered, centers_difference

def rotate_centered_field(field, angle, img_size):
    field_rot = kornia.geometry.transform.rotate(
        tensor=field.unsqueeze(1), angle=angle.reshape(-1), center=img_size/2, padding_mode="border"
        )
    return field_rot


def translate_field(field, centers_difference, img_size):

    # swap x and y entries because of the nature of kornia translate
    translation = centers_difference.flip(dims=(-2, -1))
    field_translated = kornia.geometry.transform.translate(
        tensor=field.unsqueeze(1), translation=translation[:, :, 0] * img_size[0],
        padding_mode="border"
        )
    return field_translated

