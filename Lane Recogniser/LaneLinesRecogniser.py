import torch
import torch.nn as nn
import torch.functional as F

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import random


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
#print(subdirs)

"""def delete_image(image_dir):
    for img in image_dir:
        os.remove(img)

for folder in [1,1,1,1]:
    if folder != 20:
        delete_image(folder)"""


#Test that the lane lines are drawn in the correct place
sample_img_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set/clips/0313-1/6040/20.jpg"
sample_seg_label_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set/seg_label/0313-1/6040/20.png"
sample_label_data_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set/label_data_0313.ndjson"

#Function takes in the ndjson and the line, then creates the correct directory for the image and seg_label 
def match_video_to_lane_data(lane_df, line): 
    raw_file = lane_df["raw_file"][line]
    folder = "SELF DRIVING CAR/DATA/TUSimple/train_set/"
    img_dir = os.path.join(folder, raw_file)
    raw_file = raw_file[6:] #Removes the "clips" part of the raw file dir
    raw_file = raw_file[0:len(raw_file)-3] #Removes jpg, since the seg label is a png!
    raw_file = raw_file + "png"
    seg_label_dir = os.path.join(folder, "seg_label/", raw_file)
    return img_dir, seg_label_dir

def draw_precalced_lines(lane_df, label_data_line: int): 
    img_dir, seg_label_dir = match_video_to_lane_data(lane_df, label_data_line) 
    img = np.asarray(Image.open(img_dir))
    seg_label = np.asarray(Image.open(seg_label_dir))
    seg_label = cv2.convertScaleAbs(seg_label, 1.5, 10) #Increases constrast and brightness

    lane_x_coords = np.array(lane_df["lanes"][label_data_line]) #List of x-coordinates of the lanes
    lane_y_coords = np.array(lane_df["h_samples"][label_data_line]) #List of y-coordinates of the lanes
    #Shape of lane_x and lane_y -> (4, 48) (48,)

    fig, ax = plt.subplots()
    ax.imshow(img)
    colors = ["r","r","b","g"] #In the x_coords 4D array, the first two arrays are the immediate lane lines 
    #and then 3rd is the one to the left and the 4th is the one to the right (there is max 4 dims in the x_coords)
    for j, color in enumerate(colors): 
        for i in range(0, len(lane_y_coords)):
            if lane_x_coords[j][i] == -2: #x = -2 indicates a non value.
                continue
            else:
                print(lane_x_coords[j][i], lane_y_coords[i])
                ax.scatter(x = lane_x_coords[j][i], y = lane_y_coords[i], 
                            c = color, s = 20)
    plt.show()
    
#Example
r = random.randint(0,2858)
lane_data = pd.read_json(sample_label_data_dir, lines=True)
draw_precalced_lines(lane_data, r) #Careful, since 2858 is the length of 0313 label data


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

#Processing image requires: resize, grayscale, GaussianBlur, region of interest
#Leaving out hough transform at first as I want to see whether the CNN can make curved lines without me having
#to specifically code it in using the hough transform or a variant of it
    
def grayscale(img): #Reduced complexity in greyscale
    img = img.resize() #Find what size we need
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img): #Takes an input using the Gaussian distribuiton of the pixel colour around the target pixel
    return cv2.GaussianBlur(img)

def region_of_interest(img): #Take a triangular or trapezium-shaped mask of the image
    height, width = img.shape 
    triangle = np.array([[(0, height), (650, 150), (width, height)]])
    mask = np.zeros_like(img) #Creates black image with the same dimensions as original image
    mask = cv2.fillPoly(mask, triangle, 255) #255 is white (hence why the numpy array is 0 everywhere else)
    mask = cv2.bitwise_and(img, mask) #Bitwise_and finds the parts of the image which are similar and different and groups them
    return mask

def process_image(img_dir):
    img = np.asarray(Image.open(img_dir))
    gray = grayscale(img)
    gauss = gaussian_blur(gray)
    processed_img = region_of_interest(gauss)
    plt.imshow(processed_img)
    return processed_img

process_image()


class CNN(nn.Module):
    def __init__(self):
        super.__init__()
        pass

    def forward():
        pass


