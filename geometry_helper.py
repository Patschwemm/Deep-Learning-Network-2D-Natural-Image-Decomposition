import torch
import primitives
import matplotlib.pyplot as plt    
from typing import Tuple


def distance_field(x: Tuple[int, int, int, int, y: torch.Tensor])

def tensor_to_rectangle(x: torch.Tensor, y: torch.Tensor):
    """ 
    input x Tensor is of shape [int, int, int, int] for x-center, y-center, length and width of rectangle.
    input rectangle and shape of image and create a mask of 0 where the empty space is
    and 1 where the rectangle area is. Edge case is that it cuts off the rectangle once it is outside the image.
    """
    center = x[0:2]
    length = x[2]
    width = x[3]


    print(y.shape)

    x_min = center[0] - torch.div(width, 2, rounding_mode="trunc")
    x_min = torch.clamp(x_min, min=0, max=255).type(torch.DoubleTensor)
    
    x_max = center[0] + torch.div(width, 2, rounding_mode="trunc")
    x_max = torch.clamp(x_max, min=0, max=255).type(torch.DoubleTensor)

    y_min = center[1] - torch.div(length, 2, rounding_mode="trunc")
    y_min = torch.clamp(y_min, min=0, max=255).type(torch.DoubleTensor)

    y_max = center[1] + torch.div(length, 2, rounding_mode="trunc")
    y_max = torch.clamp(y_max, min=0, max=255).type(torch.DoubleTensor)

    mask = torch.zeros(y.shape)
    mask[int(x_min.item()) : int(x_max.item()) , int(y_min.item()) : int(y_max.item())] = 1
    return mask



def intersection_area(x: torch.Tensor, y: torch.Tensor, output_tensor = False):
    """ 
    Computes the intersections Tensor for two tensors and the sum of that.
    Requires the GT to be 1 for objects and 0 for background.
    """

    # get intersect mask
    intersect = torch.logical_and(x, y)
    plt.imshow(intersect, cmap="gray")
    plt.savefig("pictures/bin_mask.png")
    plt.show()
    plt.close()
    # sum up how many pixels are intersected
    sum = torch.sum(intersect)
    return sum if output_tensor == False else sum, intersect

def union_area(x: torch.Tensor, y: torch.Tensor, output_tensor = False):
    """ 
    Computes the union Tensor for two tensors and the sum of that.
    Requires the GT to be 1 for objects and 0 for background.
    """

    # get union mask
    union = torch.logical_or(x, y)
    sum = torch.sum(union)
    return sum if output_tensor == False else sum, union



    return prim_binary_mask

def pred_to_primitive(center_x: int, center_y: int, length: int, width: int):

    return primitives.Rectangle(center = [center_x, center_y], length=length, width=width)


x = torch.Tensor([128, 128, 100, 50], requires_grad=True)
y = torch.ones((256, 256))
rect = tensor_to_rectangle(x, y)
print(intersection_area(rect, y, output_tensor=False))
print(union_area(rect, y, output_tensor=False))


