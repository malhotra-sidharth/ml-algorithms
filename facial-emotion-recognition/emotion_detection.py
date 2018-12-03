# references
# https://www.superdatascience.com/opencv-face-detection/
import pandas as pd
import os
import torch
import torchvision
from facecrop import CropFace

# extract images names from folder
images = os.listdir('./data/images')

# get the emotions of each image
labels = pd.read_csv('./data/legend.csv')
# print(labels['emotion'][labels['image'] == images])

# save cropped images
# UNCOMMENT BELOW LINES TO RECROP ALL IMAGES
# cf = CropFace()
# cf.crop_multiple_images(images, './data/images')







