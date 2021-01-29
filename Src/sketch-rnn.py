"""
Imports Section
"""

import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import random
random.seed(0)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.autograd import Variable

from torchvision.utils import save_image
import cv2
import os
import torchvision
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

"""
Auto-Enconder Class

Need to fit in the VGG encoder model and Sketch-RNN decoder model
"""

image_size = 244 # this is dependent on what you model takes e.g. vgg takes 224

class AutoEncoder(Module):
    def __init__(self,bottleneck_size):
        super(AutoEncoder, self).__init__()
        # INITIALIZE YOUR TRAINING PARAMETERS HERE.

    def encoder(self,image):
        # WRITE ENCODER ARCHITECTURE HERE or USE PYTORCH MODELS

        return code
    
    def decoder(self,code):
        # WRITE DECODER ARCHITECTURE HERE
        
        return decoded_image
    
    def forward(self,image):
        # PUT IT TOGETHER HERE
        code = self.encoder(image)
        decoded_image = self.decoder(code)
        return decoded_image

"""
Data Loader Section
"""

class Chairs(Dataset):
    # IMPLEMENT THIS DATA LOADING CLASS
    def __init__(self, dataset_path=""):
        # DEFINE YOUR PARAMETERS AND VARIABLES YOU NEED HERE.
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
        self.input_imgs_names = sorted(os.listdir(dataset_path))
        self.input_imgs_paths = [os.path.join(dataset_path, name) for name in self.input_imgs_names]
        
    def __len__(self):
        # RETURN SIZE OF DATASET
        length = len(self.input_imgs_names)
        return length

    # cv2 transforms data into tensors
    def __getitem__(self, idx):
        # RETURN IMAGE AT GIVEN idx
        # use cv2 to load images
        img = cv2.imread(self.input_imgs_paths[idx])
        input_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        input_image = self.transform(input_image)
        return input_image

"""
Variables needed for loading data and training

"""
batch_size = 20
chairs_train_data_path = "../datasets/quickdraw/chair/train"
chairs_train_dataset = Chairs(chairs_train_data_path)
torch_train_chairs = DataLoader(chairs_train_dataset, shuffle=True, batch_size=batch_size, num_workers=1)

chairs_val_data_path = "../datasets/quickdraw/chair/val"
chairs_val_dataset = Chairs(chairs_val_data_path)
torch_val_chairs = DataLoader(chairs_val_dataset, shuffle=True,batch_size=batch_size,num_workers=1)

"""
Training Section
"""

def reconstruction_loss(producedSketch, desiredSketch):
    # L2 RECONSTRUCTION LOSS
    criterion = nn.MSELoss()
    return criterion(producedSketch, desiredSketch)

epochs = 3 # CHOOSE YOUR EPOCH SIZE TO GET BEST RESULTS

# May not need this if using pytorch model
chairs_bottleneck_size = 128 # CHOOSE YOUR BOTTLENECK SIZE. 

model_chairs_vgg = AutoEncoder(chairs_bottleneck_size)

# Optimizer section, using Adam
chairs_optimizer = optim.Adam(model_chairs_vgg.parameters(), lr=1e-3) 

train_loss = []
for ep in range(epochs):
    running_loss = 0.0
    for i,data in enumerate(torch_train_chairs):
        # This is dependent on your pytorch model specification

        # reshape mini-batch data to [N, 784] matrix so it can be loaded
        data = data.view(-1, data.shape[1]*data.shape[2]*data.shape[3])
        
        # resetting the gradients back to zero, pytorch accumulates gradients each backward pass
        chairs_optimizer.zero_grad()
        
        # encoder layer pass, compute the reconstructions
        new_img = model_chairs_vgg(data)
        
        # compute loss to optimize reconstruction with
        loss = reconstruction_loss(new_img, data)
        
        # back propagation
        loss.backward()
        
        # updtae parameters
        chairs_optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        running_loss += loss.item()
    
    loss = running_loss / 100 # 100 images
    
    #train_loss.append(loss)
    print(f'Epoch {ep+1} of {epochs}, Train Loss: {loss:.5f}')

"""
Testing Section
"""

# Choose 5 random images to recreate and see if a recreation of the image looks like a chair
chairs_random_images = random.sample(list(chairs_val_dataset),5)
chairs_loader = DataLoader(chairs_random_images,shuffle=True,batch_size=1,num_workers=0)

# Plot the sketches using matplotlib

for i, data in enumerate(torch_val_chairs):
    img_shape = data.shape
    data = data.view(-1, data.shape[1]*data.shape[2]*data.shape[3])
    outputs = model_chairs_vgg(data)

    f, axarr = plt.subplots(1,2)
    
    inp_image_tensor = data.reshape(img_shape)
    inp_image = inp_image_tensor.squeeze(0).detach().cpu().numpy()
    inp_image = inp_image[0, :,:,:].transpose(1,2,0)
    
    out_image_tensor = outputs.reshape(img_shape)
    out_image = out_image_tensor.squeeze(0).detach().cpu().numpy()
    out_image = out_image[0, :,:,:].transpose(1,2,0)

    axarr[0].imshow(inp_image)
    axarr[1].imshow(out_image)
    if i == 4: break

"""
Save Model Section
- this is so we can compare all our models afterwards with ned input sketches
"""

PATH = "../models/vgg_autoencoder.pt" #change this {chosen encode}_autoencoder.pt
torch.save(model_chairs_vgg.state_dict(), PATH)


