# vgg16_research_project
Implemented a VGG-16 model and tested on data of MRI brain scans

#@title Load your dataset { display-mode: "form" }
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from keras.preprocessing.image import ImageDataGenerator

!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20%2B%20X/Group/Healthcare/Brain%20Tumor%20Detection/tumor.npy"
!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20%2B%20X/Group/Healthcare/Brain%20Tumor%20Detection/tumor_labels.npy"

### pre-loading all data of interest
image_data = np.load('tumor.npy')
labels = np.load('tumor_labels.npy')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision import transforms

tumorcount = 0
nontumorcount = 0
for i in range(0,253):
  if labels[i] == 1:
    tumorcount += 1
  else:
    nontumorcount += 1

print("Tumors: " + str(tumorcount))
print("Not-tumors: " + str(nontumorcount))

normalized_images = []
for im in image_data:
  normalized_images.append((im - np.min(im))/(np.max(im)-np.min(im)))
normalized_images = np.array(normalized_images)

# Define the transformations
mean = image_data.mean()
std = image_data.std()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=mean, std=std),
    transforms.ToTensor(),
])
image_data_transformed = transform(image_data)
# Convert arrays to tensors and apply transformations
image_data_transformed = [transform(img) for img in image_data]
image_data_tensor = torch.stack(image_data_transformed)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Combine into a single dataset
dataset = TensorDataset(image_data_tensor, labels_tensor)

# Split the dataset into training and testing sets
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# # Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class VGG16(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16, self).__init__()

        # Define the VGG16 blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create the VGG16 model
vgg16_model = VGG16(num_classes=2)
optimizer = optim.Adam(vgg16_model.parameters(), lr=0.002)
print(vgg16_model)

from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, SubsetRandomSampler

def train_model(model, train_dataset, optimizer, criterion, num_epochs=100, k_folds=5):

  kfold = KFold(n_splits=k_folds, shuffle=True)

  # Check if CUDA is available and set the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = model.to(device)

    # Start the k-fold cross-validation
  for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(train_dataset)))):

    # Print
    print(f"FOLD {fold}")
    print("--------------------------------")

    # Create train and validation datasets using indices
    train_sampler = SubsetRandomSampler([i for i in train_ids])
    val_sampler = SubsetRandomSampler([i for i in val_ids])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0   # to store number of correct predictions
        num_in_batch = 0     # to store total number of samples

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > .5).float()
            num_in_batch += labels.size(0)
            #correct += predicted.eq(labels).sum().item()
            correct += (predicted == labels).sum().item()
        epoch_accuracy = 100. * correct / num_in_batch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                val_predicted = (outputs > .5).float()
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()
        val_epoch_accuracy = 100. * val_correct / val_total
        val_epoch_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%")






    # Validation for this fold
    # model.eval()
    # val_loss = 0.0
    # val_correct = 0
    # val_total = 0
    # with torch.no_grad():
    #     for i, (inputs, labels) in enumerate(val_loader):
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         labels = labels.unsqueeze(1).float()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()

    #         val_predicted = (outputs > .5).float()
    #         val_total += labels.size(0)
    #         val_correct += (val_predicted == labels).sum().item()

    # # Print validation results for this fold
    # print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {100. * val_correct / val_total}%")
    # print("--------------------------------")

# Define loss function and optimizer
criterion = nn.BCELoss()
vgg16_trained_model = train_model(vgg16_model, train_dataset, optimizer, criterion, num_epochs = 20)
