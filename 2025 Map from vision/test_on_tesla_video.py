#test on real Tesla video
# copy from past tensorflow code - need to change to pytorch

import cv2 #to work with images from cameras
import numpy as np #in this example to change image representation - re-shaping

from matplotlib import pyplot as plt #to show images in this notebook

import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn


import torchvision.transforms as transforms
from PIL import Image


SIZE_X = 640
SIZE_Y = 480

m3_video = cv2.VideoCapture('G:/My Drive/Self-driving/real tesla footage.mp4')


# define reverse disctionary to go from one-hot to colours
color_to_label_mapping = {
    0: (0, 0, 0),  #Nothing
    1: (128, 64, 128),  #road
    2: (157, 234, 50),  #lane marking
    3: (0, 0, 142),     #cars
    4: (220, 220, 0)    #road signs
}

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        dec4 = self.dec4(F.interpolate(bottleneck, scale_factor=2))
        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2))
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2))
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2))
        
        return self.final(dec1)

def convert_to_rgb(image):
    image = (np.squeeze(image[0].numpy()))

    rgb_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.int32)
    #pick class with max probability
    rgb_img = np.argmax(image,axis=2)
    rgb_img = np.array([[color_to_label_mapping[value] for value in row] for row in rgb_img])
    return rgb_img

transform = transforms.Compose([
    transforms.Resize((SIZE_Y, SIZE_X)),  # Resize to match model input size
    transforms.ToTensor(),           # Convert to tensor
    
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device',device)
model = UNet().to(device) 

#load a trained model
state_dict = torch.load('C://SelfDrive//2025 Map from vision//unet_model_20250224_1.pth')  
model.load_state_dict(state_dict)


while(m3_video.isOpened()):
    # Capture frame-by-frame
    ret, frame = m3_video.read()
    if ret == True:
        # Display the resulting frameabs
        frame = cv2.resize(frame, (SIZE_X,SIZE_Y))
        
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = Image.fromarray(input_frame)
        input_frame = transform(input_frame)
        input_frame = input_frame.unsqueeze(0) # add extra batch dim 

        with torch.no_grad():  # Disable gradient computation for inference
            predicted = model(input_frame.to(device))  #predict
        
        # Convert output tensor to image
        predicted_sem_img = predicted.squeeze(0).detach().cpu()
        predicted_sem_img = transforms.ToPILImage()(predicted_sem_img)

        predicted_sem_img = np.array(predicted_sem_img)

        predicted_sem_img = cv2.cvtColor(predicted_sem_img, cv2.COLOR_BGR2RGB)

        # Resize predicted_map to match rgb_im
        #predicted_sem_img = cv2.resize(predicted_sem_img, (rgb_im.shape[1], rgb_im.shape[0]))  # Resize to (W, H)



        #cv2.imshow('predicted semantic',img_rgb)
        im_h = cv2.hconcat([frame,predicted_sem_img])
        cv2.imshow('2 cameras', im_h)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
        # Break the loop
    else: 
        break
# When everything done, release the video capture object
m3_video.release()
 
# Closes all the frames
cv2.destroyAllWindows()