import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #choose whether to use gpu or cpu

#hyper parameters
num_epochs = 30
batch_size = 1
learning_rate = 0.0005

# def get_data():
data_dir = 'C:\Users\Public\PartIIB project 2023_2024\Image collection without reaction\00AgNO3_mole_fraction\Outputs_Grayscale_Labelled_Images_Sizes\size_folder' #path to data

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  #transforms data to tensor and normalizes it

# full_set = datasets.ImageFolder(data_dir, transform=transform) #loads training data, need to figure out how to link image to labels, probably using the data files
# train_set, val_set, test_set = random_split(full_set, [int(len(full_set)*0.75), int(len(full_set)*0.15), int(len(full_set)*0.1)]) #splits data into training, validation and test sets

train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module): # note need to find out image size
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) #in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(5,5) #kernel_size, stride (shift x pixel to the right)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(16*4*4, 120) # 5x5 is the size of the image after 2 conv layers, 16 is the number of channels, 120 is the number of nodes in the hidden layer
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 16*4*4)  #flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train):
        # 100, 1, 28, 28
        # 100, 784
        images = images.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
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
    n_correct = 0
    n_samples = 0
    for images, labels in test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc}')
#Data set is split into training and test sets (85% and 15%)