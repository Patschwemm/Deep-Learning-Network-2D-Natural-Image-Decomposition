# load the pet dataset, needs to have dataset according to readme loaded

from tqdm import tqdm 
from os import listdir
import numpy as np
from matplotlib import image


def load_images():
    png_list = listdir("dataset/images")

    data_list = []
    print("Loading Images")
    for png in tqdm(png_list):
        if (png.endswith(".mat")):
            continue
        img_data = image.imread("dataset/" + "images/" + png)
        data_list.append(img_data)

    data_list = np.asarray(data_list, dtype=object)

    return data_list



