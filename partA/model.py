import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import traceback
import argparse
import wandb
from torchinfo import summary


class FlexibleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, num_filters=[32,32,32,32,32], filter_size=3, 
                 activation_fn=nn.ReLU, dense_neurons=128, num_blocks=5, drop_out=0, batch_normalization="No"):
        super(FlexibleCNN, self).__init__()

        self.conv_blocks = nn.ModuleList()
        self.num_blocks = num_blocks
        dimension = 224
        
        for block in range(num_blocks):
            in_channel = num_filters[block]
            if (block == 0):
                in_channel = input_channels
            else:
                in_channel = num_filters[block-1]

            layers = []
            # Convolutional layer
            layers.append(nn.Conv2d(in_channel, num_filters[block], kernel_size=filter_size, padding=1))
            
            # Batch normalization (if enabled)
            if batch_normalization=="Yes":
                layers.append(nn.BatchNorm2d(num_filters[block]))
            
            # Activation function
            layers.append(activation_fn())
            
            # Dropout (if enabled)
            if drop_out > 0:
                layers.append(nn.Dropout2d(drop_out))
            
            # Max pooling
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            self.conv_blocks.append(nn.Sequential(*layers))
            dimension /= 2
        
        dimension = int(dimension)
        self.flatten_size = num_filters[-1] * dimension * dimension
        
        # Fully connected layers with batch norm and dropout
        self.fc1 = nn.Linear(self.flatten_size, dense_neurons)
        self.bn_fc1 = nn.BatchNorm1d(dense_neurons) if batch_normalization == "Yes" else nn.Identity()
        self.activation = activation_fn()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()
        self.fc2 = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):

        for block in self.conv_blocks:
            x = block(x)

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with batch norm and dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x