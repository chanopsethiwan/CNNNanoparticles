import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #choose whether to use gpu or cpu

#hyper parameters
num_epochs = 30
batch_size = 1
learning_rate = 0.0005

#custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(os.listdir(root_dir))
        self.labels = [self.extract_label(img) for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def extract_label(self, img_name):
        # Assuming that the label is part of the filename before the first underscore
        label = float(img_name[-17:-5])
        return label

# def get_data():
data_dir = r'C:\Users\Public\PartIIB project 2023_2024\Image collection without reaction\00AgNO3_mole_fraction\Outputs_Grayscale_Labelled_Images_Sizes\size_folder' #path to data

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5), (0.5))])  #transforms data to tensor and normalizes it

full_set = CustomImageDataset(root_dir=data_dir, transform=transform) #loads data from directory and applies transform
train_set, val_set, test_set = random_split(full_set, [int(len(full_set)*0.75), int(len(full_set)*0.15), int(len(full_set)*0.100056)]) #splits data into training, validation and test sets

train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module): # note need to find out image size
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,8,10, padding='same') #in_channels, out_channels, kernel_size
        self.normalise1 = nn.BatchNorm2d(8)
        # self.pool = nn.MaxPool2d(5,5) #kernel_size, stride (shift x pixel to the right)
        self.pool1 = nn.AvgPool2d(10, stride=10)
        self.conv2 = nn.Conv2d(8, 16, 10, padding='same')
        self.normalise2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 10, padding='same')
        self.normalise3 = nn.BatchNorm2d(32) 
        self.conv4 = nn.Conv2d(32, 32, 10, padding='same')
        # self.fc1 = nn.Linear(16*3*3, 120) # 3x3 is the size of the image after 2 conv layers, 16 is the number of channels, 120 is the number of nodes in the hidden layer
        # self.fc2 = nn.Linear(120,84)
        # self.fc3 = nn.Linear(60, 1)
        self.fc = nn.Linear(32*5*5, 1)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.pool1(F.relu(self.normalise1(self.conv1(x)))) 
        x = self.pool2(F.relu(self.normalise2(self.conv2(x)))) 
        x = F.relu(self.normalise3(self.conv3(x)))
        x = F.relu(self.normalise3(self.conv4(x)))
        x = x.view(-1, 32*5*5)  #flatten
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

model = ConvNet().to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train):
        images = images.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        labels = labels.float()
        loss = criterion(outputs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 ==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

print("Finished Training")

#test 
#need to change testing to be based on continuous value
with torch.no_grad(): # no need to calculate gradient
    squared_difference = 0
    for images, labels in test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        #value, index
        _, predictions = torch.max(outputs, 1)
        squared_difference += (predictions - labels) ** 2
    
    rmse = torch.sqrt(squared_difference / len(test))
    print(f'RMSE = {rmse}')

#Data set is split into training and test sets (85% and 15%)