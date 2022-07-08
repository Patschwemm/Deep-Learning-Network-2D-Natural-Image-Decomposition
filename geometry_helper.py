import torch
import matplotlib.pyplot as plt    
from typing import Tuple


def distance_field_rectangle(x: Tuple[int, int, int, int], y: torch.Tensor):
    """ 
    x is in form x[0] = x_center, x[1] = y_center, x[2] = length, x[3] = width.
    This Function computes the distance field according to:
    C_rect = (|p_x| - length)^2_+ + (|p_y| - width)^2_+
    _+ meanng max(0, x)
    """
    
    axis_x = torch.linspace(start=0, end=y.shape[0]-1, steps = y.shape[0])
    axis_y = torch.linspace(start=0, end=y.shape[1]-1, steps = y.shape[1])

    grid_x, grid_y = torch.meshgrid(axis_x, axis_y, indexing="ij")

    # calculate the border of the rectangle
    xborderleft = x[0] - x[3]/2
    xborderright = x[0] + x[3]/2
    yborderup = x[1] - x[2]/2
    yborderdown = x[1] + x[2]/2

    # compute distances for dimensions x and y
    c_x = torch.maximum( xborderleft - grid_x, grid_x - xborderright )
    c_y = torch.maximum( yborderup - grid_y, grid_y - yborderdown )

    # if a point is inside the rectangle, default to 0 distance
    # this is the case if the maximum of both distances equl to zero
    c_x = torch.maximum(torch.zeros(c_x.shape), c_x)
    c_y = torch.maximum(torch.zeros(c_y.shape), c_y)

    # merge distance of dimensions to obtain field distances
    c_field_distance = torch.sqrt(c_x**2 + c_y**2)
    print("penner")

    mask = c_field_distance == 0



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

    return c_field_distance
    

def distance_field_batch(x: torch.Tensor, y_shape: torch.Tensor, device: torch.device):

    w, h = y_shape[2], y_shape[1]


    # bbox_test = torch.tensor([
    # # [(P1_x, P1_y), (P2_x, P2_y), ...]
    # [(128 - 12, 128 - 12), (128 + 12, 128 + 12)],
    # [(128 - 12, 128 - 12), (128 + 12, 128 + 12)]
    # ]).float()


    # coordinate system
    X = torch.arange(w).to(device)
    Y = torch.arange(h).to(device)
    # reshape to match bounding boxes
    X, Y = X[None, :, None], Y[None, :, None]
    # compute L1-distances in each dimension to both boundaries
    Dx, Dy = x[:, None, :, 0] - X, x[:, None, :, 1] - Y
    Dx[..., 1], Dy[..., 1] = Dx[..., 1] * -1, Dy[..., 1] * -1
    # compute 
    Dx, Dy = Dx.max(dim=-1).values, Dy.max(dim=-1).values
    Dx = torch.maximum(torch.zeros_like(Dx), Dx)
    Dy = torch.maximum(torch.zeros_like(Dy), Dy)
    D = Dx.unsqueeze(-1)**2 + Dy.unsqueeze(-2)**2

    # plt.imshow(D[0] == 0, cmap='gray')
    # plt.savefig("pictures/dist_field_batched.png")
    # plt.show()
    # plt.close()

    return D


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

def pred_to_primitive(center_x: int, center_y: int, length: int, width: int):

    return primitives.Rectangle(center = [center_x, center_y], length=length, width=width)


# x = torch.tensor([
#     [128., 128., 100., 50.],
#     [100, 100, 50 ,20],
#     [80, 80, 30, 60]
#     ])
# y = torch.ones((256, 256))
# dist = distance_field_batch(x, y.shape)

# rect = tensor_to_rectangle(x, y)
# print(intersection_area(rect, y, output_tensor=False))
# print(union_area(rect, y, output_tensor=False))


