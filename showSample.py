import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import os

def sample_single():
    dfile = "./samples_4x256x256x3.npz"
    data = np.load(dfile)
    os.mkdir("./results/model_09/")

    for i in range(data['arr_1'].shape[0]):
        print(data['arr_0'][i].shape)
        image = Image.fromarray(data['arr_0'][i])
        image.save(f'./results/model_09/image_{i}.jpg')

sample_single()