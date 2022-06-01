# load the pet dataset, needs to have dataset according to readme loaded

import numpy as np
from tqdm import tqdm 
from os import listdir
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def load_images(samples, original_img=False):
    png_list = listdir("dataset/images")

    data_list = []
    original_list = []
    print("Loading Images")
    for png in tqdm(png_list[:samples]):
        if (png.endswith(".mat")):
            continue
        
        with Image.open("dataset/" + "images/" + png) as img:
            if (original_img == True):
                original_list.append(img)
            img = np.asarray(img.resize((224, 224), 1).convert("RGB"))
            data_list.append(img/255)


    data_list = np.asarray(data_list, dtype=np.double)
    data_list = torch.as_tensor(data_list, dtype=torch.float64)


    return (data_list, original_list) if original_img == True else data_list



