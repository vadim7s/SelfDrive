'''
continue from torch_model2.py

try RGB to semantic, but in pytorch this time

Result: just 1 epoch over 200k images achieved great results when testing trained model
on a real Tesla footage 

'''
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.transforms import ToPILImage
import cv2


# Custom dataset class
class MapDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        
        input_image = transform(input_image)
        target_image = transform(target_image)

        return input_image, target_image

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

# Define a function to save the predicted image
def save_predicted_image(tensor_in,tensor_out, epoch,batch, output_dir='C://SelfDrive//2025 Map from vision//prediction_eg_sem'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert tensor to PIL Image
    to_pil = ToPILImage()
    img_in = to_pil(tensor_in.squeeze(0).cpu())  # Squeeze batch dimension and move to CPU
    img_out = to_pil(tensor_out.squeeze(0).cpu())  # Squeeze batch dimension and move to CPU
    
    # Save as PNG
    img_in.save(os.path.join(output_dir, f"epoch_{epoch+1}_{batch}_train.png"))
    img_out.save(os.path.join(output_dir, f"epoch_{epoch+1}_{batch}_predicted.png"))


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


# Define generator to yield data in batches
def data_generator(dataloader):
    while True:
        for inputs, targets in dataloader:
            yield inputs, targets

# Paths to input and target images
input_dir = 'C://SelfDrive//2025 Map from vision//img'
target_dir = 'C://SelfDrive//2025 Map from vision//sem_img'

# Dataset and DataLoader
dataset = MapDataset(input_dir, target_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device',device)
model = UNet().to(device) #original starty a new model
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)


#load last saved model
#state_dict = torch.load('C://SelfDrive//2025 Map from vision//unet_model_20250111.pth')  
#model.load_state_dict(state_dict)


# Training loop
num_epochs = 1
data_gen = data_generator(dataloader)

for epoch in range(num_epochs):
    print('Starting to train epoch ',epoch)
    model.train()
    epoch_loss = 0
    
    for _ in range(len(dataloader)):
        
        inputs, targets = next(data_gen)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if _ % 100 ==0:
            print(f'processed batch {_} out of {len(dataloader)} ... epoch loss {loss.item():.4f} epoch {epoch+1}')
            if _ % 1000 ==0:
                # save an example of a predicted image ad the end of each epoch
                save_predicted_image(inputs[0],outputs[0], epoch,_)


    # update stats
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
    torch.save(model.state_dict(), f'C://SelfDrive//2025 Map from vision//unet_model_20250224_{epoch+1}.pth')