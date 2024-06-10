import torch
import torch.nn as nn
import torch.functional as F

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import pandas as pd


"""
This project consists of the main steps:

- Create custom dataset of labelled lane line images of a road
- CNN model (saved) to interpret and learn from these images
- Pipeline to process images 
- CNN to then predict the outcome of a new image
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set"
test_data_dir = "SELF DRIVING CAR/DATA/TUSimple/test_set"

"""
Data is images from a video shot at 20fps, so will delete 19 out of 20 of the images and rename the image
to the name of the folder then take the image out of the folder and delete the empty folder.

But then this messes up the order of the json files and also unsure about which image corresponds the seg_label
image.

Turns out the end of the json ends "raw_file = 20.jpg" so this means we can possibly clear out all of the 
extra images? Or use them for testing purposes?
"""

subdirs = [x[0] for x in os.walk("SELF DRIVING CAR/DATA/TUSimple/train_set/clips")]
print(subdirs)

def delete_image(image_dir):
    for img in image_dir:
        os.remove(img)

for folder in [1,1,1,1]:
    if folder != 1:
        delete_image(folder)


#Test that the lane lines are drawn in the correct place
sample_img_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set/clips/0313-1/60/20.jpg"
sample_seg_label_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set/seg_label/0313-1/60/20.png"
sample_label_data_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set/label_data_0313.ndjson"

def draw_precalced_lines(img_dir, seg_label_dir, label_data_dir, label_data_line):
    img = np.asarray(Image.open(img_dir))
    seg_label =  np.asarray(Image.open(seg_label_dir))
    plt.imshow(img)
    plt.show()

    seg_label = cv2.convertScaleAbs(seg_label, 1.5, 10) #Increases constrast and brightness
    plt.imshow(seg_label)
    plt.show()

    lane_data = pd.read_json(label_data_dir, lines=True)

    #with open(label_data_dir, "r") as file:
        #lane_data = json.load(file)
    lane_x_coords = lane_data["lanes"][label_data_line] #List of x-coordinates of the lanes
    lane_y_coords = lane_data["h_samples"][label_data_line] #List of y-coordinates of the lanes
    raw_file = lane_data["raw_file"][label_data_line] 

    print(lane_x_coords.shape,lane_y_coords.shape)

    for i in range(0, len(lane_x_coords)):
        plt.plot(lane_x_coords[i],lane_y_coords[i])
    plt.imshow(img)
    plt.show()
    

    
draw_precalced_lines(sample_img_dir,sample_seg_label_dir, sample_label_data_dir, 0)


class LaneLinesDataset(Dataset):
   
    def __init__(self, root_dir, transform = None):
        """
        Arguments:
        root_dir (string) -> contains all of the image files
        """

    def __len__():
        pass

    def __get_item__():
        
        pass

#Processing image requires: resize, grayscale, GaussianBlur, hough transform
def grayscale(img):
    img = img.resize() #Find what size we need
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img):
    return cv2.GaussianBlur(img)

def hough_transform(img):
    pass

def process_image(img):
    gray = grayscale(img)
    gauss = gaussian_blur(gray)



class CNN():
    def __init__(self):
        super.__init__()
        pass

    def forward():
        pass


