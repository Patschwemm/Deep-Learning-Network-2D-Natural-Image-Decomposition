import torch
import numpy as np
import torchvision.transforms as T
from skfmm import distance
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
from losses import *
from matplotlib import pyplot as plt
import typing as t
from typing import Tuple
from typing_extensions import Literal
import cv2
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


def distance_field_dataset_extraction(train_dataset: torch.utils.data.Dataset):
    imgs = []
    gt_masks = []
    distance_fields = []
    print("Extracting images, GT masks and distance fields")
    for img, target in tqdm(train_dataset):
        imgs.append(img)
        gt_masks.append(target[0])
        distance_fields.append(target[1])

    imgs = torch.stack(imgs)
    gt_masks = torch.stack(gt_masks)
    distance_fields = torch.stack(distance_fields)

    return imgs, gt_masks, distance_fields


class PILToTensor_for_targets:
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # to make a binary mask, set gray(2) to 0 and black and white pet to 1
        target[(target == 1) | (target == 3)] = 1
        target[target == 2] = 0
        # target = scipy.ndimage.median_filter(target, size=(3,3))
        target = target[None, :, :]
        return target


class Distance_field_for_targets:
    def __call__(self, target):
        if torch.max(target) == 1:
            try:
                distance_field = torch.from_numpy(distance(phi=-target + 0.5, dx=1))
                distance_field = distance_field.masked_fill(target, 0)  
            except:
                distance_field = torch.zeros_like(target)
        else:
            distance_field = torch.zeros_like(target)

        if type(target) != type(distance_field):
            print("type not the same")
            print("target: ", type(target))
            print("field: ", type(distance_field))

        return torch.stack((target, distance_field)).float()

class binary_mask_mnist:
    def __call__(self, target):
        target = (target > 0) * 1
        return target.float()


class OxfordIIITPet_Distancefields_train(OxfordIIITPet):
    def __init__(self):

        # transform like preprocess required by the network
        tf_rgb = T.Compose([T.ToTensor(), T.Resize(256), T.CenterCrop(256),])

        tf_gray = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(256),
                PILToTensor_for_targets(),
                Distance_field_for_targets(),
            ]
        )

        super().__init__(
            root="dataset",
            split="trainval",
            target_types="segmentation",
            transform=tf_rgb,
            target_transform=tf_gray,
            download=True,
        )


class OxfordIIITPet_Distancefields_test(OxfordIIITPet):
    def __init__(self):

        # transform like preprocess required by the network
        tf_rgb = T.Compose([T.ToTensor(), T.Resize(256), T.CenterCrop(256),])

        tf_gray = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(256),
                PILToTensor_for_targets(),
                Distance_field_for_targets(),
            ]
        )

        super().__init__(
            root="dataset",
            split="test",
            target_types="segmentation",
            transform=tf_rgb,
            target_transform=tf_gray,
            download=True,
        )


class Rectangle_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        rect_amount: int = 1, 
        circ_amount: int = 1,
        samples: int = 1000, 
        img_size: torch.tensor=torch.tensor([256, 256]),
        mode: str="random"
        ) -> None:
        super().__init__()

        # generate multiple rectangles of shape (samples, 2, 2)
        multiple_rects, rot_list = self.generate_rectangles(samples, rect_amount, mode)
        # generate multipe circles of shape (samples, 1, 3)
        multiple_circles = self.generate_circles(samples, circ_amount, mode)

        # reshape for distance field function
        rects = torch.tensor([multiple_rects])
        circs = torch.tensor([multiple_circles])
        rots = torch.tensor(rot_list).squeeze()
        rects_reshaped = torch.reshape(rects, (samples, rect_amount, 2, 2))
        circs_reshaped = torch.reshape(circs, (samples, circ_amount, 1, 3))


        # use distance field functions in order to create distancefield,
        dist_fields = []
        masks_x = []
        for i in range(samples):
            # get rectangle union distance field
            if rect_amount > 0:
                rect_field = compute_rotated_rectangle_distance_field(rects_reshaped[i], rots[i],  img_size) 
                rect_field = rect_field.min(dim=0).values
            else:
                rect_field = torch.ones(size=(img_size[0], img_size[1])).unsqueeze(0) * float("inf")
            # get circle union distance field
            if circ_amount > 0:
                circ_field = compute_circle_distance_field(circs_reshaped[i], img_size)
                circ_field = circ_field.min(dim=0).values
            else: 
                circ_field = torch.ones(size=(img_size[0], img_size[1])) * float("inf")

            # take the union of both fields
            union_field = torch.cat((rect_field, circ_field.unsqueeze(0)), dim=0)
            union_field = (union_field.min(dim=0).values)

            mask = (union_field == 0)

            dist_fields.append(union_field)
            masks_x.append(mask)


        dist_fields = torch.stack(dist_fields).unsqueeze(0)
        masks_x = torch.stack(masks_x).unsqueeze(0)
        self.data = torch.cat([masks_x, dist_fields], dim=0)
        
        # _, ax = plt.subplots(1, 2, figsize=(20, 20))
        # ax[0].imshow(masks_x[0], cmap="gray")
        # ax[1].imshow(union_dist_field[0], cmap="gray")
        # plt.savefig(f"pictures/synthethic_rects{0}.png")
        # plt.close()

    def generate_rectangles(self, samples: int, rect_amount:int, mode: str):
        # create variable amount of rectangles with one corner and the diagonal
        # opposite corner
        multiple_rects = []
        multiple_rotations = []
        for i in range(samples):
            rect_list = []
            rotations = []
            for amount in range(rect_amount):
                if mode == "random":
                    randoms = sorted([random.random() for j in range(4)])
                    rect_list.append(randoms)
                    rotations.append([torch.tensor([random.random() * 361]) for j in range(4)])
                elif mode == "hardcoded":
                    if amount == 0:
                        randoms = [0.25, 0.75, 0.25, 0.75]
                        rect_list.append(randoms)
                        rotations.append(torch.tensor([0.0]))
                    elif amount == 1:
                        randoms = [0.6, 0.8, 0.1, 0.5]
                        rect_list.append(randoms)
                        rotations.append(torch.tensor([135.0]))
                    elif amount == 2:
                        randoms = [0.1, 0.5, 0.6, 0.95]
                        rect_list.append(randoms)
                        rotations.append(torch.tensor([270.0]))
                    elif amount == 3:
                        randoms = [0.6, 0.95, 0.6, 0.95]
                        rect_list.append(randoms)
                        rotations.append(torch.tensor([295.0]))
            multiple_rotations.append(rotations)
            multiple_rects.append(rect_list)
        return multiple_rects, multiple_rotations

    def generate_circles(self, samples: int, circ_amount:int, mode: str):
    # create variable amount of rectangles with one corner and the diagonal
    # opposite corner
        multiple_circs = []
        for i in range(samples):
            circ_list = []
            for amount in range(circ_amount):
                if mode == "random":
                    randoms = self.generate_circles_xyradius()
                    circ_list.append(randoms)
                elif mode == "hardcoded":
                    if amount == 0:
                        randoms = [0.5, 0.5, 0.35]
                        circ_list.append(randoms)
                    elif amount == 1:
                        randoms = [0.8, 0.8, 0.15]
                        circ_list.append(randoms)
                    elif amount == 2:
                        randoms = [0.25, 0.7, 0.2]
                        circ_list.append(randoms)
                    elif amount == 3:
                        randoms = [0.2, 0.2, 0.2]
                        circ_list.append(randoms)
            multiple_circs.append(circ_list)
        return multiple_circs

    def generate_circles_xyradius(self):
        random_center = torch.tensor([random.random() for j in range(2)])
        distance_to_border_x = torch.min(torch.abs(1-random_center[0]), random_center[0])
        distance_to_border_y = torch.min(torch.abs(1-random_center[1]), random_center[1])
        total_min_distance = torch.min(distance_to_border_x, distance_to_border_y)
        random_radius = (random.random() * total_min_distance/2)
        return [random_center[0].item(), random_center[1].item(), random_radius.item()]


    def __getitem__(self, idx):
        return self.data[0, idx], self.data[1, idx]

    def __len__(self):
        return len(self.data[0])


def mask_rgb_imgs(x, y):
    y_masks = torch.permute(y, (1, 0, 2, 3, 4))[0]
    y_masks.expand(x.shape)
    x_masked = y_masks * x
    return x_masked

def get_simple_2d_transforms() -> t.Callable[
    [t.Union[Image.Image, np.ndarray]], torch.Tensor
]:
    return T.Compose([T.Resize(64), T.Grayscale(), T.ToTensor()])


def process_single_2d_image(
    image: np.ndarray,
    transforms: t.Optional[
        t.Callable[[np.ndarray], t.Union[torch.Tensor, np.ndarray]]
    ],
) -> t.Tuple[torch.Tensor, ...]:
    height, width = image.shape[0], image.shape[1]
    current_max_distance = max(width, height) * math.sqrt(2)
    thresholded = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0).astype(
        np.uint8
    )
    distances = cv2.distanceTransform(1 - thresholded, cv2.DIST_L2, maskSize=0)
    coords = np.stack(
        np.meshgrid(range(width), range(height)), axis=-1
    ).reshape(
        (-1, 2)
    )  # -> N, (x, y)
    distances = distances[coords[:, 1], coords[:, 0]]
    dim = max(height, width)
    coords = coords.astype(np.float32)
    coords = (coords + 0.5) / dim - 0.5
    if transforms is not None:
        image = Image.fromarray(image, mode="RGB")
        image = transforms(image)
    else:
        image = torch.from_numpy(image).float() / 255
    coords = torch.from_numpy(coords).float()
    distances = (distances <= 0).astype(np.float32)
    return image, coords, distances

class CADDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        data_split: Literal["train", "valid", "test"],
        transforms: t.Optional[t.Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.transforms = transforms
        self.data_split = data_split
        if data_split == "train":
            self.data_key = "train_images"
        elif data_split == "valid":
            self.data_key = "val_images"
        else:
            self.data_key = "test_images"
        with h5py.File(self.h5_file_path, "r") as h5_file:
            self._images = h5_file[self.data_key][:]
        self.__cache = {}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        if index in self.__cache:
            return self.__cache[index]
        image = self._images[index].astype(np.uint8) * 255
        image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
        image, coords, distances = process_single_2d_image(
            image, self.transforms
        )
        self.__cache[index] = (image, coords, distances)
        return image, image

def dataloader(
        training: bool, split_type: Literal["train", "valid"], num_workers
    ) -> DataLoader:
        data_path = './dataset/CAD/cad.h5'
        batch_size = 128
        transforms = get_simple_2d_transforms()
        loader = DataLoader(
            dataset=CADDataset(data_path, split_type, transforms),
            batch_size=batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=num_workers,
        )
        return loader

def cad_train_dataloader(num_workers) -> DataLoader:
    return dataloader(True, "train", num_workers)
def cad_val_dataloader(num_workers) -> DataLoader:
    return dataloader(False, "valid", num_workers)


