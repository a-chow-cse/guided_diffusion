import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import os

dfile = "/tmp/openai-2023-04-07-08-00-35-808664/samples_10x128x128x3.npz"
data = np.load(dfile)
os.mkdir("./results/model_05/")

for i in range(data['arr_1'].shape[0]):
    print(data['arr_0'][i].shape)
    image = Image.fromarray(data['arr_0'][i])
    image.save(f'./results/model_05/image_{i}.jpg')