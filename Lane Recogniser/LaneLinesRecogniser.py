import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import random
from skimage import io
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


"""
(American spellings used in code from now on to reduce error and confusion, also have switched to using double quotes in text and 
single quotes in parameters/identifiers)

This script is the first step in the self driving car project, it uses the TUSimple dataset, which consists of labelled images
of a road, with coordinates of the lane lines. 
These images are processed and then fed through a CNN, for the network to then learn where the lane lines should be.
The outcome of the project should be that the script can identify the correct lane lines on a road and output a visual.
"""


train_data_dir = "SELF DRIVING CAR/DATA/TUSimple/train_set"
test_data_dir = "SELF DRIVING CAR/DATA/TUSimple/test_set"

"""
Wrote a function which deletes all buy image 20, then rename the file and delete 
the old folders. ie 0313-1/60/20.png becomes 60.png
"""

def clean_folder_structure(root_dir): #Eg. this will be .../0313-1
    for subdir in os.listdir(root_dir): #Eg. will be 60, 120, 180, ...
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            target_file = os.path.join(subdir_path, "20.jpg") #Adds a slash where commas are!

            try:
                if os.path.exists(target_file):
                    subdir_jpg = subdir + ".jpg"
                    new_name = os.path.join(root_dir, subdir_jpg)
                    shutil.move(target_file, new_name)
                    shutil.rmtree(subdir_path) #Deletes old folder with 19 images in
                else:
                    pass

            except FileNotFoundError as e:
                print(f"Error: {e}")
                
"""
Next three functions used to change the directories of the "raw_files" part of the ndjson file, so that the excess
folders could be removed in the main data folders
"""
def load_ndjson_to_dataframe(ndjson_file):
    with open(ndjson_file, "r") as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

def save_dataframe_to_ndjson(lane_df, output_file):
    with open(output_file, "w") as file:
        for record in lane_df.to_dict(orient="records"):
            file.write(json.dumps(record) + "\n")

def rename_raw_files(lane_df, json_file_path):
    for index, row in lane_df.iterrows():
        raw_file_path = row["raw_file"] #Old example: "raw_file": "clips/0313-1/60/20.jpg"
        #The name should be: "clips/0313-1/60.jpg" So essentially removing the "/20" for every raw_file name
        new_filename = raw_file_path[:-7]
        new_raw_file_path = new_filename + ".jpg"
        lane_df.at[index, "raw_file"] = new_raw_file_path

        only_json_name = os.path.basename(json_file_path)
        save_dataframe_to_ndjson(lane_df, only_json_name)
    
"""
Only needed to run this once to renmame everything, if more data is added from TUSimple ds, then will be useful

json_file_list = ["SELF DRIVING CAR/DATA/TUSimple/train_set/label_data_0313.ndjson",
                  "SELF DRIVING CAR/DATA/TUSimple/train_set/label_data_0531.ndjson",
                  "SELF DRIVING CAR/DATA/TUSimple/train_set/label_data_0601.ndjson",
                  "SELF DRIVING CAR/DATA/TUSimple/test_set/test_label.ndjson"]

for json_file in json_file_list:
    lane_df = load_ndjson_to_dataframe(json_file)
    rename_raw_files(lane_df, json_file)

"""
    
root_dir_train = "SELF DRIVING CAR/DATA/TUSimple/train_set/clips"
root_dir_test = "SELF DRIVING CAR/DATA/TUSimple/test_set/clips"

for subdir_ in os.listdir(root_dir_train):
    train_data_subfolder = os.path.join(root_dir_train, subdir_)
    if os.path.isdir(train_data_subfolder):
        clean_folder_structure(train_data_subfolder)

for subdir_ in os.listdir(root_dir_test):
    test_data_subfolder = os.path.join(root_dir_test, subdir_)
    if os.path.isdir(test_data_subfolder):
        clean_folder_structure(test_data_subfolder)

#Function takes in the ndjson and the line, then creates the correct directory for the image and seg_label 
def match_video_to_lane_data(lane_df, line, seg_label_boolean): 
    raw_file = lane_df["raw_file"][line]
    folder = "SELF DRIVING CAR/DATA/TUSimple/train_set/"
    img_dir = os.path.join(folder, raw_file)
    raw_file = raw_file[6:] #Removes the "clips" part of the raw file dir
    raw_file = raw_file[0:len(raw_file)-3] #Removes jpg, since the seg label is a png!
    raw_file = raw_file + "png"
    seg_label_dir = os.path.join(folder, "seg_label/", raw_file)
    if seg_label_boolean == True:
        return img_dir, seg_label_dir
    else:
        return img_dir

def draw_precalced_lines(lane_df, label_data_line: int): #Killing off the excess lanes for now. (changes colors to 4 colors to revert)
    img_dir, seg_label_dir = match_video_to_lane_data(lane_df, label_data_line, True) 
    img = np.asarray(Image.open(img_dir))
    seg_label = np.asarray(Image.open(seg_label_dir))
    seg_label = cv2.convertScaleAbs(seg_label, 1.5, 10) #Increases constrast and brightness

    lane_x_coords = np.array(lane_df["lanes"][label_data_line]) #List of x-coordinates of the lanes
    lane_y_coords = np.array(lane_df["h_samples"][label_data_line]) #List of y-coordinates of the lanes
    #Shape of lane_x and lane_y -> (4, 48) (48,)

    fig, ax = plt.subplots()
    ax.imshow(img)
    colors = ["b","b"] #In the x_coords 4D array, the first two arrays are the immediate lane lines 
    #and then 3rd is the one to the left and the 4th is the one to the right (there is max 4 dims in the x_coords)
    for j, color in enumerate(colors): 
        for i in range(0, len(lane_y_coords)):
            if lane_x_coords[j][i] == -2: #x = -2 indicates a non value.
                continue
            else:
                ax.scatter(x = lane_x_coords[j][i], y = lane_y_coords[i], 
                            c = color, s = 20)
    plt.show()

#Processing image requires: resize, grayscale, GaussianBlur, region of interest
#Leaving out hough transform at first as I want to see whether the CNN can make curved lines without me having
#to specifically code it in using the hough transform or a variant of it
    
def grayscale(img): #Reduced complexity in greyscale
    #img = img.resize() #Find what size we need
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img): #Takes an input using the Gaussian distribuiton of the pixel colour around the target pixel
    return cv2.GaussianBlur(img, (5,5), 1) #Tuple is the kernel of the image, higher the int (SD) is the more blur

def region_of_interest(img): #Take a triangular or trapezium-shaped mask of the image
    height, width = img.shape 
    triangle = np.array([[(0, height), (650, 150), (width, height)]])
    mask = np.zeros_like(img) #Creates black image with the same dimensions as original image
    mask = cv2.fillPoly(mask, triangle, 255) #255 is white (hence why the numpy array is 0 everywhere else)
    mask = cv2.bitwise_and(img, mask) #Bitwise_and finds the parts of the image which are similar and different and groups them
    return mask

def process_image(df: pd.DataFrame, json_line: int) -> np.array: #Debugging currently, this function is not the culprit for the max = min = 1 error
    try:
        raw_file_path = df[json_line]['raw_file'] #clips/60.jpg
    except KeyError:
        print(f'Error: Index {json_line} out of range for DataFrame.')
        return None
    
    img_dir = os.path.join("SELF DRIVING CAR/DATA/TUSimple/train_set", raw_file_path)
    img = cv2.imread(img_dir)
    #plt.imshow(img)
    #plt.show()

    if img is None:
        print(f'Error loading image: {img_dir}')
        return None
    
    gray = grayscale(img)
    #plt.imshow(gray)
    #plt.show()
    gauss = gaussian_blur(gray)
    #plt.imshow(gauss)
    #plt.show()
    normalise = gauss / 255.0  #Put the values between 0 and 1
    #plt.imshow(normalise)
    #plt.show()
    processed_img = region_of_interest(normalise)
    #plt.imshow(processed_img)
    #plt.show()
    processed_img = np.array(processed_img, dtype=np.float32)  
    return processed_img

class LaneLinesDataset(Dataset):
    def __init__(self, ndjson_file, root_dir, transform = None, max_coords = 96): 
        with open(ndjson_file, 'r') as file:
            self.coordinates = [json.loads(line) for line in file] #self.coordinates is a pandas dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.max_coords = max_coords #96 length (as 48 coords in each line for x)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        img = process_image(self.coordinates, index) #Inputs df and index, outputs 2D np array of the image
        
        if img is None:
            return None

        x_coords = (np.array(self.coordinates[index]['lanes']))
        x_coords = x_coords[:2].flatten() #Only taking the immidiate lane lines
        y_coords = np.array(self.coordinates[index]['h_samples'])
        y_coords = (np.vstack([y_coords, y_coords])).flatten()

        spliced_coords = np.array([])
        
        for idx, val in enumerate(y_coords):
            spliced_coords = np.append(spliced_coords, (x_coords[idx], val)) #Should create an array of coordinates
        spliced_coords = spliced_coords.flatten()

        if len(spliced_coords) > self.max_coords: #Give standard length for spliced_coords, may to be be changed 
            spliced_coords = spliced_coords[:self.max_coords]
        else:
            spliced_coords = np.pad(spliced_coords, (0, self.max_coords - len(spliced_coords)), 'constant', constant_values=-1)

        spliced_coords = spliced_coords.astype(np.float32) #Error "found dtype double when expected float"

        if self.transform:
            img = self.transform(img)

        return img, spliced_coords

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  #Resize image to 224x224 as per an error message
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
])

pytorch_lane_dataset = LaneLinesDataset(ndjson_file="SELF DRIVING CAR/DATA/TUSimple/train_set/train_label_data.ndjson",
                                   root_dir="SELF DRIVING CAR/DATA/TUSimple/train_set/clips",
                                   transform=transform)


#Input an image and with the image comes one array of coordinates where the lines are.
#Want to process this image into a 2D array and identify the road lines.
#Then want the coordinates of these road lines, then to check them against the pre-determined ones.

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #1 because grayscale, 6 output feature maps, 5 kernel
        self.pool = nn.MaxPool2d(2, 2) #Pooling layer uses a 2x2 to get max value
        self.conv2 = nn.Conv2d(6, 16, 5) #6 input channels because previously had 6 outputs, 16 output feature maps
        self.conv3 = nn.Conv2d(16, 32, 5) #Added a third conv layer and pooling layer to see if the loss decreased, however didn't affect it. (may need to remove later)

        self.fc1 = nn.Linear(32 * 24 * 24, 120) #16 outputs *5^2, 120 output features - should be changed
        self.fc2 = nn.Linear(120, 84) #120 input channels, 84 output
        self.fc3 = nn.Linear(84, 96) #84 input channels, 96 output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) #Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""
The next two functions are AI generated and will be removed in the final version, time restrictions and I wanted to attempt to
see the results of the CNN, however this was not possible, but highlighted the important error in need of fixing
"""
def visualize_processed_image(processed_img):
    # Convert processed image from float32 to uint8 for visualization
    processed_img = (processed_img * 255).astype(np.uint8)

    # Convert grayscale image to RGB for visualization with matplotlib
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(processed_img_rgb, cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()

"""with open("SELF DRIVING CAR/DATA/TUSimple/train_set/train_label_data.ndjson", 'r') as file:
            df = [json.loads(line) for line in file] 
processed_img = process_image(df, 1)
visualize_processed_image(processed_img)"""

def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    samples = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            for i in range(inputs.size(0)):
                if samples >= num_samples:
                    return

                # Convert tensor image to numpy array
                img = inputs[i].cpu().numpy().transpose((1, 2, 0))  # Convert to HWC format
                img = img * 255  # Denormalize (since it was normalized by dividing by 255)
                img = img.astype(np.uint8)

                print(f"Image shape: {img.shape}, min: {img.min()}, max: {img.max()}")  # Debugging print

                # Ensure image is in correct range
                img = np.clip(img, 0, 255)

                # Handle grayscale images
                if img.shape[2] == 1:  # If image is grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # Ensure coordinates are within image bounds
                img_height, img_width = img.shape[:2]
                pred_coords = outputs[i].cpu().numpy().reshape(-1, 2)
                true_coords = labels[i].cpu().numpy().reshape(-1, 2)

                pred_coords = np.clip(pred_coords, 0, [img_width, img_height])
                true_coords = np.clip(true_coords, 0, [img_width, img_height])

                # Draw the predicted coordinates in red
                for coord in pred_coords:
                    cv2.circle(img, (int(coord[0]), int(coord[1])), 3, (0, 0, 255), -1)

                # Draw the true coordinates in green
                for coord in true_coords:
                    if coord[0] != -2:  # Assuming -2 is an invalid value and should not be plotted
                        cv2.circle(img, (int(coord[0]), int(coord[1])), 3, (0, 255, 0), -1)

                # Convert BGR to RGB for displaying with matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                plt.figure()
                plt.imshow(img_rgb)
                plt.title("Red: Predicted, Green: True")
                plt.axis('off')
                plt.show()

                samples += 1




if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cnn = CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    train_loader = DataLoader(pytorch_lane_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(pytorch_lane_dataset, batch_size=64, shuffle=True, num_workers=2)

    train_losses = []
    val_losses = []

    for epoch in range(2): 
        cnn.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: 
                print(f'[Epoch {epoch + 1}, Mini-batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        train_losses.append(running_loss / len(train_loader))

        #Calculate validation loss
        cnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.3f}')

    print("Finished Training")

    #Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    visualize_predictions(model = cnn, dataloader = train_loader, device = device, num_samples=1)

"""
Massive loss at the moment (17,000), not sure what is causing it possibly:
    - Not enough images (2000 training)
    - Learning rate?

Actually have plotted the images and the similarity between the predicted and actual lines is very similar,
so I think that the loss could definatley further be optimised however, 17000 is actually not as bad as it sounds.
Only one lane line is being shown at the moment, as well as the image is not being displayed correctly
in the visualize_predictions function. Due to the min and max color of the image both being 1. Need to fix 
these issue to properly visualise the results of the CNN.
"""
