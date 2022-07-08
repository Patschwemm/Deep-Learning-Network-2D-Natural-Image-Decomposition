import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from typing import Dict


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
                (rect[i, j, 2] - rect[i, j, 0]),
                (rect[i, j, 3] - rect[i, j, 1]),
            ]

            # create a rectangle patch
            fig_rect = patches.Rectangle(
                (rect[i, j, 0:2]),
                rect_length[0],
                rect_length[1], 
                linewidth=1,
                edgecolor=edgecolors[i],
                facecolor="none"
            )
            axis.add_patch(fig_rect)

    plt.show()

