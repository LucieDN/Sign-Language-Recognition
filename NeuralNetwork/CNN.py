
import numpy as np
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
print(f"x_train: {X_training.shape}. y_train: {Y_training.shape}")

# Labels, i.e. fashion categories associated to images (one category per image)
sign_labels = {
    0: "AUSSI",
    1: "LS",
    2: "OUI",
}

learning_rate = 1e-3
n_epochs = 10
batch_size = 64

sign_train_dataloader = DataLoader(list(zip(X_training, Y_training)), batch_size=batch_size)
sign_test_dataloader = DataLoader(list(zip(X_test, Y_test)), batch_size=batch_size)


class Convnet(nn.Module):
    """Convnet for fashion articles classification"""

    def __init__(self):
        super().__init__()

        # Define a sequential stack
        self.layer_stack = nn.Sequential(
            # Feature extraction with convolutional and pooling layers
            nn.Conv2d(in_channels=9540, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Classification with fully connected layers
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Compute output of layer stack
        logits = self.layer_stack(x)

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits


sign_convnet = Convnet().to(device)
print(sign_convnet)

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
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

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


fashion_history = train_fashion(
    sign_train_dataloader,
    sign_convnet,
    # Standard loss for multiclass classification
    nn.CrossEntropyLoss(),
    # Adam optimizer for GD
    optim.Adam(sign_convnet.parameters(), lr=learning_rate),
)















