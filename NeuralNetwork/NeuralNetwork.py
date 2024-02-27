import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from torch import nn, optim
from sklearn.calibration import column_or_1d
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

import torch
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
# Convertir les données en tableaux NumPy si ce n'est pas déjà fait
X_np = X_training.astype(np.float32)  # Assurez-vous que les données sont au format float32 pour les tenseurs PyTorch
Y_np = Y_training.astype(np.int64)  # Assurez-vous que les étiquettes sont au format int64 pour les tenseurs PyTorch


# Create PyTorch tensors from NumPy tensors
X_training = torch.tensor(X_training, dtype=torch.float32).to(device)
Y_training = torch.tensor(Y_training, dtype=torch.long).to(device)
print(f"x_train: {X_training.shape}. y_train: {Y_training.shape}")
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)
print(f"x_train: {X_test.shape}. y_train: {Y_test.shape}")


# Labels, i.e. fashion categories associated to images (one category per image)
sign_labels = {
    0: "AUSSI",
    1: "LS",
    2: "OUI",
}

# Try to change the learning rate to 1e-2 ans check training results
learning_rate = 1e-3
n_epochs = 10
batch_size = 64
inputs = 9540
outputs = 3

sign_train_dataloader = DataLoader(list(zip(X_training,Y_training)), batch_size=batch_size)
sign_test_dataloader = DataLoader(list(zip(X_test,Y_test)), batch_size=batch_size)


# for X_batch, Y_batch in sign_train_dataloader:
#     print(X_batch)

class NeuralNetwork(nn.Module):
    """Neural network for fashion articles classification"""

    def __init__(self):
        super().__init__()

        # Flatten the input image of shape (1, 28, 28) into a vector of shape (28*28,)
        self.flatten = nn.Flatten()

        # Define a sequential stack of linear layers and activation functions
        self.layer_stack = nn.Sequential(
            # First hidden layer with 784 inputs
            nn.Linear(in_features=inputs, out_features=64),
            nn.ReLU(),
            # Second hidden layer
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            # Output layer
            nn.Linear(in_features=64, out_features=outputs),
        )

    def forward(self, x):
        """Define the forward pass of the model"""

        # Apply flattening to input
        x = self.flatten(x)

        # Compute output of layer stack
        logits = self.layer_stack(x)

        # Logits are a vector of raw (non-normalized) predictions
        # This vector contains 10 values, one for each possible class
        return logits
    
sign_model = NeuralNetwork().to(device)
print(sign_model)
# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(sign_model.parameters(), lr=learning_rate)


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

def train_sign(dataloader, model, loss_fn, optimizer):
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


sign_history = train_sign(
    sign_train_dataloader,
    sign_model,
    loss_fn,
    optimizer,
)


def plot_loss_acc(history):
    """Plot training loss and accuracy. Takes a Keras-like History object as parameter"""

    loss_values = history["loss"]
    recorded_epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(recorded_epochs, loss_values, ".--", label="Training loss")
    ax1.set_ylabel("Loss")
    ax1.legend()

    acc_values = history["acc"]
    ax2.plot(recorded_epochs, acc_values, ".--", label="Training accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.legend()

    final_loss = loss_values[-1]
    final_acc = acc_values[-1]
    fig.suptitle(
        f"Training loss: {final_loss:.5f}. Training accuracy: {final_acc*100:.2f}%"
    )
    plt.show()


plot_loss_acc(sign_history)


# Mettre le modèle en mode d'évaluation
sign_model.eval()

# Passer les données de test dans le modèle pour obtenir les logits
logits = sign_model(X_test)
# Appliquer la fonction Softmax pour obtenir les probabilités pour chaque classe
probabilities = nn.functional.softmax(logits, dim=1)
print("Ok")
print(probabilities[0])

# Obtenir les classes prédites en choisissant l'indice de la classe ayant la probabilité la plus élevée
predicted_classes = torch.argmax(probabilities, dim=1)

# Convertir les tenseurs PyTorch en tableaux NumPy
predicted_classes = predicted_classes.cpu().numpy()

# Afficher les prédictions
for i, predicted_class in enumerate(predicted_classes):
    print(f"Exemple {i+1}: Classe prédite - {sign_labels[predicted_class]}, Classe réelle : {sign_labels[int(Y_test[i])]} ")

# Évaluer les performances du modèle
from sklearn.metrics import accuracy_score

# Convertir les données de test en tableaux NumPy si ce n'est pas déjà fait
X_test_np = X_test.cpu().numpy()
Y_test_np = Y_test.cpu().numpy()

# Calculer la précision du modèle
accuracy = accuracy_score(Y_test_np, predicted_classes)
print(f"Précision du modèle : {accuracy * 100:.2f}%")
