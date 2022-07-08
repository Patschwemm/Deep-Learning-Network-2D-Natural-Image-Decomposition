import torch
import matplotlib.pyplot as plt
from typing import Tuple 
import random
import numpy as np
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt
import utils
from skfmm import distance
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from tqdm import tqdm



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

    def __call__(self,  target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # to make a binary mask, set gray(2) to 0 and black and white pet to 1
        target[(target == 1) | (target == 3)] = 1
        target[target == 2] = 0
        #target = scipy.ndimage.median_filter(target, size=(3,3))
        target = target[None, :, :]
        return target

class Distance_field_for_targets:

    def __call__(self, target):
        if torch.max(target) == 1:
            distance_field = torch.from_numpy(distance(phi=-target + 0.5, dx=1))
            distance_field = distance_field.masked_fill(target, 0)
        else:
            distance_field = torch.zeros_like(target)
        
        if (type(target) != type(distance_field)):
            print("type not the same")
            print("target: ", type(target))
            print("field: ", type(distance_field))

        return torch.stack((target, distance_field)).float()

class OxfordIIITPet_Distancefields_train(OxfordIIITPet):

    def __init__(self):

        # transform like preprocess required by the network
        tf_rgb = T.Compose([
            T.ToTensor(),
            T.Resize(256),
            T.CenterCrop(256),
        ])

        tf_gray = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            PILToTensor_for_targets(),
            Distance_field_for_targets()
        ])

        super().__init__(root = "dataset", split="trainval", target_types="segmentation", 
                               transform=tf_rgb, target_transform=tf_gray, download=True)

class OxfordIIITPet_Distancefields_test(OxfordIIITPet):

    def __init__(self):

        # transform like preprocess required by the network
        tf_rgb = T.Compose([
            T.ToTensor(),
            T.Resize(256),
            T.CenterCrop(256),
        ])

        tf_gray = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            PILToTensor_for_targets(),
            Distance_field_for_targets()
        ])

        super().__init__(root = "dataset", split="trainval", target_types="segmentation", 
                               transform=tf_rgb, target_transform=tf_gray, download=True)


def mask_rgb_imgs(x,y):
    y_masks = torch.permute(y,(1, 0, 2, 3, 4))[0]
    y_masks.expand(x.shape)
    x_masked = y_masks * x
    return x_masked

        
    #     imgs, gt_masks, distance_fields = distance_field_dataset_extraction(train_dataset)

    #     self.imgs = imgs
    #     self.gt_masks = gt_masks
    #     self.distance_fields = distance_fields

    # def __getitem__(self, index):
    #     return (self.imgs[index], self.gt_masks[index], self.distance_fields[index])

    # def __len__(self):
    #     return len(self.imgs)
    
# k = len(gt_mask)

# distance_fields = -torch.stack([
#     # compute signed distance field for current mask
#     torch.from_numpy(distance(
#         phi=-mask + 0.5,
#         dx=1
#     ))
#     # loop over all masks
#     for mask in tqdm(gt_mask[:k])
#     # fill interior with zeros
# ]).masked_fill(gt_mask[:k], 0.0)

