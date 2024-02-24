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
import joblib

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


def count_parameters(model, trainable=True):
    """Return the total number of (trainable) parameters for a model"""

    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable
        else sum(p.numel() for p in model.parameters())
    )

X_training = np.array(joblib.load("DataManipulation/Data/X_training.pkl"))
Y_training = np.array(joblib.load("DataManipulation/Data/Y_training.pkl"))
X_test = np.array(joblib.load("DataManipulation/Data/X_test.pkl"))
Y_test = np.array(joblib.load("DataManipulation/Data/Y_test.pkl"))

# Rate of parameter change during gradient descent
learning_rate = 0.1
# An epoch is finished when all data samples have been presented to the model during training
n_epochs = 50
# Number of samples used for one gradient descent step during training
batch_size = X_training.shape[0]//3
# Number of neurons on the hidden layer of the MLP
hidden_layer_size = 2
in_features = X_training.shape[1]

# Create PyTorch tensors from NumPy tensors
x_train = torch.from_numpy(X_training).float().to(device)

# PyTorch loss function expects float results of shape (batch_size, 1) instead of (batch_size,)
# So we add a new axis and convert them to floats
y_train = torch.from_numpy(Y_training[:, np.newaxis]).float().to(device)
print(f"x_train: {x_train.shape}. y_train: {y_train.shape}")

dataloader = DataLoader(list(zip(X_training, Y_training)), batch_size=batch_size)

# Binary cross entropy loss function
loss_fn = nn.BCELoss()

class NeuralNetwork(nn.Module):
    """Neural network for fashion articles classification"""

    def __init__(self):
        super().__init__()

        # Flatten the input image of shape (1, 28, 28) into a vector of shape (28*28,)
        self.flatten = nn.Flatten()

        # Define a sequential stack of linear layers and activation functions
        self.layer_stack = nn.Sequential(
            # First hidden layer with 784 inputs
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            # Second hidden layer
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            # Output layer
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Apply flattening to input
        x = self.flatten(x)

        # Compute output of layer stack
        logits = self.layer_stack(x.float())

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits

def softmax(x):
    """Softmax function"""

    return np.exp(x) / sum(np.exp(x))

def epoch_loop(dataloader, model, loss_fn, optimizer):
    """Training algorithm for one epoch"""

    total_loss = 0
    n_correct = 0

    for x_batch, y_batch in dataloader:
        # Load data and targets on device memory
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x_batch.float())
        loss = loss_fn(output.squeeze(), y_batch.type(torch.LongTensor))


        # Backward pass: backprop and GD step
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Accumulate data for epoch metrics: loss and number of correct predictions
            total_loss += loss.item()
            n_correct += (model(x_batch).argmax(dim=1) == y_batch).float().sum().item()

    return total_loss, n_correct

def train_fashion(dataloader, model, loss_fn, optimizer):
    """Main training loop"""

    history = {"loss": [], "acc": []}
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)

    print(f"Training started! {n_samples} samples. {n_batches} batches per epoch")

    for epoch in range(n_epochs):
        total_loss, n_correct = epoch_loop(dataloader, model, loss_fn, optimizer)

        # Compute epoch metrics
        epoch_loss = total_loss / n_batches
        epoch_acc = n_correct / n_samples

        print(
            f"Epoch [{(epoch + 1):3}/{n_epochs:3}]. Mean loss: {epoch_loss:.5f}. Accuracy: {epoch_acc * 100:.2f}%"
        )

        # Record epoch metrics for later plotting
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

    print(f"Training complete! Total gradient descent steps: {n_epochs * n_batches}")

    return history

model = NeuralNetwork().to(device)
print(model)

fashion_history = train_fashion(
    dataloader,
    model,
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(), lr=learning_rate),
)


