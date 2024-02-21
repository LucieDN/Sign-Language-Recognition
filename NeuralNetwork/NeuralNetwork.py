# A faire avant :
# pip install numpy as np
# pip install tensorflow[and-cuda]

import platform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn
from sklearn.datasets import make_circles

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# PyTorch device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA GPU {torch.cuda.get_device_name(0)} found :)")
# Performance issues exist with MPS backend
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("MPS GPU found :)")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU instead")

# Rate of parameter change during gradient descent
learning_rate = 0.1

# An epoch is finished when all data samples have been presented to the model during training
n_epochs = 50

# Number of samples used for one gradient descent step during training
batch_size = 5

# Number of neurons on the hidden layer of the MLP
hidden_layer_size = 2

# Create PyTorch tensors from NumPy tensors

x_train = torch.from_numpy(planar_data).float().to(device)

# PyTorch loss function expects float results of shape (batch_size, 1) instead of (batch_size,)
# So we add a new axis and convert them to floats
y_train = torch.from_numpy(planar_targets[:, np.newaxis]).float().to(device)

print(f"x_train: {x_train.shape}. y_train: {y_train.shape}")

# Load data as randomized batches for training
planar_dataloader = DataLoader(
    list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True
)


