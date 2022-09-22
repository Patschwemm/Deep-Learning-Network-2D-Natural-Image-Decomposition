import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from field_geometry import * 
from typing import Dict, Tuple

def rectangle_on_img(prim_dict: Dict, img: torch.tensor, rect: torch.tensor, mode: str="gray", sample_size: int=4):

    # print("img:", img.shape)
    # print("rect:", rect.shape)
    # print("first img", img[0, 0, 0].shape)
    # print("first rect", rect[0, 0].shape)

    # create the figure
    fig, ax = plt.subplots(1, sample_size, figsize=(20,20))
    print(img.shape)

    # display the image
    edgecolors = ["b", "r", "g", "b", "r", "g","b", "r", "g", "b", "r", "g"]

    # create the rectangle values needed
    for j, axis in enumerate(ax):
        axis.imshow(torch.permute(img[j],(1, 2, 0)))

        for i in range(prim_dict["Rectangles"]):

            rect_length = [
                abs(rect[i, j, 2] - rect[i, j, 0]),
                abs(rect[i, j, 3] - rect[i, j, 1]),
            ] * 255

            # create a rectangle patch
            fig_rect = patches.Rectangle(
                (rect[i, j, 0:2])*255,
                rect_length[0],
                rect_length[1],
                linewidth=1,
                edgecolor=edgecolors[i],
                facecolor="none"
            )
            axis.add_patch(fig_rect)
            # axis.scatter(x=rect[i, j, 0] * 255, y= rect[i, j, 1] * 255)
            # axis.scatter(x=rect[i, j, 2] * 255, y= rect[i, j, 3] * 255)

    plt.show()

def plot_rectangles(
    ax: torch.tensor, 
    r: torch.tensor, 
    img_size:Tuple[int, int]
):
    # compute rectangle 
    r = r.sort(dim=-1).values
    r = r * torch.FloatTensor(tuple(img_size))
    wh = r[:, :, 1] - r[:, :, 0]
    # display the image
    color = ["b", "r", "g"]
    for i in range(r.size(0)):
        # create a rectangle patch
        fig_rect = patches.Rectangle(
            r[i, (1, 0), 0],
            wh[i, 1],
            wh[i, 0],
            linewidth=1,
            edgecolor=color[i % len(color)],
            facecolor=color[i % len(color)],
            alpha=0.3
        )
        ax.add_patch(fig_rect)
        # create a rectangle patch
        fig_rect = patches.Rectangle(
            r[i, (1, 0), 0],
            wh[i, 1],
            wh[i, 0],
            linewidth=1,
            edgecolor=color[i % len(color)],
            facecolor='none',
            alpha=1
        )
        ax.add_patch(fig_rect)

    return ax


def plot_rotated_rectangles(
    ax: torch.tensor, 
    r: torch.tensor, 
    t: torch.tensor,
    img_size:Tuple[int, int]
):
    r = r * torch.FloatTensor(tuple(img_size))
    # rotate rectangle
    rec_all_anchors = compute_axis_aligned_rectagle_anchor_points(r)
    rec_rot = rotate_rectangle(rec_all_anchors, t)

    # display the image
    color = ["b", "r", "g"]
    for i in range(r.size(0)):
        # create a rectangle patch
        fig_rect = patches.Polygon(
            rec_rot[i, :, :].t(),
            linewidth=1,
            edgecolor=color[i % len(color)],
            facecolor=color[i % len(color)],
            alpha=0.3
        )
        ax.add_patch(fig_rect)
        # create a rectangle patch
        fig_rect = patches.Polygon(
            rec_rot[i, :, :].t(),
            linewidth=1,
            edgecolor=color[i % len(color)],
            facecolor='none',
            alpha=1
        )
        ax.add_patch(fig_rect)

    return ax

def plot_circles(
    ax: torch.tensor, 
    c: torch.tensor, 
    img_size:Tuple[int, int]
):
    # compute circle 
    c = c * img_size[0]
    # display the image
    color = ["b", "r", "g"]
    for i in range(c.size(0)):
        # create a circle patch
        fig_rect = patches.Circle(
            c[i, (1, 0)],
            c[i, 2],
            linewidth=1,
            edgecolor=color[i % len(color)],
            facecolor=color[i % len(color)],
            alpha=0.3
        )
        ax.add_patch(fig_rect)
        # create a circle patch
        fig_rect = patches.Circle(
            c[i, (1, 0)],
            c[i, 2],
            linewidth=1,
            edgecolor=color[i % len(color)],
            facecolor='none',
            alpha=1
        )
        ax.add_patch(fig_rect)

    return ax

def compute_rotation_matrix(theta):
    # compute sine and cosine
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    # build flat rotation matrix and unflatten
    R = torch.stack([cos_theta, sin_theta, -sin_theta, cos_theta], dim=-1)
    # R = torch.stack([-sin_theta, cos_theta, cos_theta, sin_theta], dim=-1)
    # R = torch.stack([sin_theta, cos_theta, cos_theta, -sin_theta], dim=-1)
    R = R.reshape(-1, 2, 2)

    return R[:, :, (1, 0)]

def compute_center_points(rec):
    return rec.mean(dim=-1, keepdims=True)

def rotate_points(points, R, p0):
    # check dimensions
    assert points.size(0) == R.size(0) == p0.size(0)
    assert points.ndim == p0.ndim
    # rotate points according to formula
    # return (R @ (points - p0).unsqueeze(-1)).reshape(p0.size())
    return p0[:, (1, 0)] + (R @ (points - p0).unsqueeze(-1)).reshape(p0.size())

def rotate_rectangle(rec, theta):
    n = rec.size(-1)
    # compute rotation matrix and centers of rotation
    # apply deg2rad to get correct theta
    R = compute_rotation_matrix(torch.deg2rad(theta))
    p0 = compute_center_points(rec)
    # repeat
    R = R.repeat_interleave(n, dim=0).reshape(-1, 2, 2)
    p0 = p0.repeat_interleave(n, dim=0).reshape(-1, 2)
    # rotate points
    p_rot = rotate_points(
        points=rec.permute(0, 2, 1).reshape(-1, 2), 
        R=R,
        p0=p0
    )
    # unravel rotated points back to rectangles
    return p_rot.reshape(-1, n, 2).permute(0, 2, 1) 

def compute_axis_aligned_rectagle_anchor_points(rec):
    assert rec.size(-1) == 2
    assert rec.size(-2) == 2

    rec = rec.sort(dim=-1).values

    rec_swap = rec.clone()
    rec_swap[:, 0, :] = rec_swap[:, 0, (1, 0)]

    return torch.stack((
        rec[:, :, 0], 
        rec_swap[:, :, 0], 
        rec[:, :, 1], 
        rec_swap[:, :, 1]
    ), dim=-1)   

def plot_rotated_rectangles_field(
    ax: torch.tensor, 
    r: torch.tensor, 
    r_rot: torch.tensor,
    img_size:Tuple[int,int]
):

    colors = ["red", "green", "blue"]
    rect_field = compute_rotated_rectangle_distance_field(r, r_rot, torch.tensor(img_size))
    mask_rect_field = (rect_field == 0) * 1

    for i in range(r.size(0)):
        gray_mask = mask_rect_field[i].squeeze()
        rgb_mask = gray_to_rgb(gray_mask, colors[i % len(colors)])
        ax.imshow(rgb_mask, alpha= 0.2)

def gray_to_rgb(mask, color):
    zeros = torch.zeros_like(mask)
    if color == "red":
        red = torch.stack([mask * 255, zeros, zeros]).permute(1, 2, 0)
        return red
    elif color == "green":
        green = torch.stack([zeros, mask * 255, zeros]).permute(1, 2, 0)
        return green
    elif color == "blue":
        blue = torch.stack([zeros, zeros, mask * 255]).permute(1, 2, 0)
        return blue
