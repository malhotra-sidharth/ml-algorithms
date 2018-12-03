# references
# https://www.superdatascience.com/opencv-face-detection/
# dataset
# https://github.com/muxspace/facial_expressions
#
import pandas as pd
import os
import torch
import torchvision
from facecrop import CropFace

# extract images names from folder
images = os.listdir('./data/images')

# get the emotions of each image
labels = pd.read_csv('./data/legend.csv')

# save cropped images
# UNCOMMENT BELOW LINES TO RE-CROP ALL IMAGES
cf = CropFace()
# cf.crop_multiple_images(images, './data/images')

# separate cropped files into respective class folders
# to be used by PyTorch Dataloader
# UNCOMMENT THIS LINE TO RE-SEPARATE THE DATASET FOR DATALOADER
cf.separate_classes_for_dataloader('./data/images/cropped', labels, images)







