import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time as time
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FashionMNIST(root="./archive", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./archive", train=False, download=True, transform=transform)
train_dataset, val_dataset = random_split(dataset, [54000, 6000])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

#CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Training Function
def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        # Save metrics
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return train_acc_list, val_acc_list, train_loss_list, val_loss_list


def test_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / np.sum(cm)
    return acc, cm


def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes,
                yticklabels=dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Train MLP
    mlp = MLP().to(device)
    startMLP = time.time()
    train_acc_mlp, val_acc_mlp, train_loss_mlp, val_loss_mlp = train_model(mlp, train_loader, val_loader, EPOCHS, LR)
    endMLP = time.time()
    # Save metrics for MLP
    print(f"MLP Training Time: {endMLP - startMLP:.2f} seconds")
    test_acc_mlp, cm_mlp = test_model(mlp, test_loader)
    with open("mlp_metrics.pkl", "wb") as f:
        pickle.dump((train_acc_mlp, val_acc_mlp, train_loss_mlp, val_loss_mlp), f)

    # Train CNN
    cnn = CNN().to(device)
    startCNN = time.time()
    train_acc_cnn, val_acc_cnn, train_loss_cnn, val_loss_cnn = train_model(cnn, train_loader, val_loader, EPOCHS, LR)
    endCNN = time.time()
    test_acc_cnn, cm_cnn = test_model(cnn, test_loader)
    # Save metrics for CNN
    with open("cnn_metrics.pkl", "wb") as f:
        pickle.dump((train_acc_cnn, val_acc_cnn, train_loss_cnn, val_loss_cnn), f)
    print(f"CNN Training Time: {endCNN - startCNN:.2f} seconds")
    print("MLP Test Accuracy:", test_acc_mlp)
    print("CNN Test Accuracy:", test_acc_cnn)
    print("MLP Val Accuracy:", sum(val_acc_mlp) / EPOCHS)
    print("CNN Val Accuracy:", sum(val_acc_cnn) / EPOCHS)
    print("MLP Train Accuracy:", sum(train_acc_mlp) / EPOCHS)
    print("CNN Train Accuracy:", sum(train_acc_cnn) / EPOCHS)
    startGraph = time.time()
    # test_acc_mlp, cm_mlp = test_model(mlp, test_loader)
    # test_acc_cnn, cm_cnn = test_model(cnn, test_loader)
    # Save Accuracy Comparison Graph
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_mlp, label="MLP Train Accuracy", color='blue')
    plt.plot(val_acc_mlp, label="MLP Val Accuracy", color='blue', linestyle='--')
    plt.plot(train_acc_cnn, label="CNN Train Acc", color='orange')
    plt.plot(val_acc_cnn, label="CNN Val Acc", color='orange', linestyle='--')
    plt.xlabel("Epoque")
    plt.ylabel("Précision")
    plt.title("Évolution de la Précision pour MLP et CNN - Validation et entraînement")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=300)
    plt.close()
    # Save MLP Loss Graph
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_mlp, label="Ent Loss")
    plt.plot(val_loss_mlp, label="Val Loss")
    plt.xlabel("Epoque")
    plt.ylabel("MSE Loss")
    plt.title("Évolution de Loss pour MLP")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "mlp_loss.png"), dpi=300)
    plt.close()
    # Save CNN Loss Grpah
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_cnn, label="Ent Loss")
    plt.plot(val_loss_cnn, label="Val Loss")
    plt.xlabel("Epoque")
    plt.ylabel("MSE Loss")
    plt.title("Évolution de Loss pour CNN")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn_loss.png"), dpi=300)
    plt.close()
    # Save Confusion Matrices Graph
    plot_confusion_matrix(cm_mlp, "Matrice de confusion pour MLP", "mlp_confusion_matrix.png")
    plot_confusion_matrix(cm_cnn, "Matrice de confusion pour CNN", "cnn_confusion_matrix.png")
    endgraph = time.time()
    print(f"Graphs saved in: {endgraph - startGraph:.2f} seconds/")
    print(f"All plots saved in: {OUTPUT_DIR}/")
