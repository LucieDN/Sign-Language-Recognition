# A faire avant :
# pip install numpy as np
# pip install tensorflow[and-cuda]

import platform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn
from sklearn.calibration import column_or_1d
from sklearn.datasets import make_circles
from sklearn.linear_model import SGDClassifier

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import joblib

# PyTorch device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA GPU {torch.cuda.get_device_name(0)} found :)")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU instead")


X_training = np.array(joblib.load("DataManipulation/Data/X_training.pkl"))
Y_training = np.array(joblib.load("DataManipulation/Data/Y_training.pkl"))
X_test = np.array(joblib.load("DataManipulation/Data/X_test.pkl"))
Y_test = np.array(joblib.load("DataManipulation/Data/Y_test.pkl"))

# Create PyTorch tensors from NumPy tensors
X_training = torch.from_numpy(X_training).float().to(device)
Y_training = torch.from_numpy(Y_training[:, np.newaxis]).float().to(device)
Y_training = column_or_1d(Y_training, warn=True)
print(f"x_train: {X_training.shape}. y_train: {Y_training.shape}")


# Using all digits as training results
#y_train = Y_training
#y_test = test_targets

# Training another SGD classifier to recognize all digits
multi_sgd_model = SGDClassifier(loss="log_loss")
multi_sgd_model.fit(X_training, Y_training)

# Since dataset is not class imbalanced anymore, accuracy is now a reliable metric
print(f"Training accuracy: {multi_sgd_model.score(X_training, Y_training):.05f}")
print(f"Test accuracy: {multi_sgd_model.score(X_test, Y_test):.05f}")




