import torch
import primitives
import matplotlib.pyplot as plt    
from typing import Tuple


def distance_field_rectangle(x: Tuple[int, int, int, int], y: torch.Tensor):
    """ 
    x is in form x[0] = x_center, x[1] = y_center, x[2] = length, x[3] = width.
    This Function computes the distance field according to:
    C_rect = (|p_x| - length)^2_+ + (|p_y| - width)^2_+
    _+ meanng max(0, x)
    """
    
    # because we are not centered in the origin, take the min distance of the two possible rectangle sides

    axis_x = torch.linspace(start=0, end=y.shape[0]-1, steps = y.shape[0])
    axis_y = torch.linspace(start=0, end=y.shape[1]-1, steps = y.shape[1])

    grid_x, grid_y = torch.meshgrid(axis_x, axis_y, indexing="ij")


    # calculate the border of the rectangle
    xborderleft = x[0] - x[3]/2
    xborderright = x[0] + x[3]/2
    yborderup = x[1] - x[2]/2
    yborderdown = x[1] + x[2]/2

    # compute field distances for dimensions x and y
    c_x = torch.maximum( xborderleft - grid_x, grid_x - xborderright )
    c_y = torch.maximum( yborderup - grid_y, grid_y - yborderdown )

    # if a point is inside the rectangle, default to 0 distance
    c_x = torch.maximum(torch.zeros(c_x.shape), c_x)
    c_y = torch.maximum(torch.zeros(c_x.shape), c_y)

    c_field_distance = torch.sqrt(c_x**2 + c_y**2)

    mask = c_field_distance == 0

    print(mask)


    # take the element wise minimum
    # c_field_distance = torch.minimum(torch.abs((x[3]/2 - grid_x + x[0]))**2 , (x[3]/2 + grid_x + x[0])**2) + torch.minimum(torch.abs((x[2]/2 - grid_y + x[1]))**2 , (x[2]/2 + grid_y + x[1])**2)
    
    plt.imshow(c_x.detach().numpy()**2, cmap="gray")
    plt.savefig("pictures/field_dist_x.png")
    plt.imshow(c_y.detach().numpy()**2, cmap="gray")
    plt.savefig("pictures/field_dist_y.png")
    plt.imshow(c_field_distance.detach().numpy(), cmap="gray")
    plt.savefig("pictures/field_dist.png")
    plt.imshow(mask, cmap="gray")
    plt.savefig("pictures/field_dist_mask")
    plt.close()

    min = torch.min(c_field_distance)

    zeros = (c_field_distance == min)
    print(zeros.nonzero(), min)
    print(x[0], x[1])


    

    


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


x = torch.tensor([128., 128., 100., 50.], requires_grad=True)
y = torch.ones((256, 256))
distance_field_rectangle(x,y)
# rect = tensor_to_rectangle(x, y)
# print(intersection_area(rect, y, output_tensor=False))
# print(union_area(rect, y, output_tensor=False))


