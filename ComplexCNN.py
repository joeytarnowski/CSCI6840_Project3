import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import regnet_x_400mf
import itertools
import pandas as pd

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 1. Dataset Loading and Visualization
def load_and_visualize_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, trainset.classes

# 2. Model Definitions
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_regnetx_model():
    model = regnet_x_400mf(weights=None)  # Initialize without pre-trained weights
    model.fc = nn.Linear(model.fc.in_features, 10)  # Modify output layer for 10 classes
    return model

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual Connection Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.residual_conv = nn.Conv2d(128, 256, kernel_size=1)  # For matching dimensions
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3 with Residual Connection
        residual = self.residual_conv(x)  # Match input dimensions
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x += residual  # Add residual connection
        x = self.pool3(x)

        # Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 3. Training and Evaluation Functions
def train_model(model, trainloader, valloader, criterion, optimizer, epochs=20):
    model = model.to(device)  # Ensure model is on the correct device
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print training loss
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(trainloader):.4f}")
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(valloader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    print('Finished Training')


def evaluate_model(model, testloader):
    model = model.to(device)  # Ensure model is on the correct device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

# 4. Hyperparameter and Model Tuning Function
def hyperparameter_tuning(param_grid):
    best_accuracy = 0.0
    best_params = None
    results = []

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in combinations:
        print(f"Testing combination: {params}")
        
        # Unpack parameters
        model_name = params['model']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        optimizer_name = params['optimizer']
        epochs = params['epochs']
        
        # Select the model
        if model_name == "SimpleCNN":
            model = SimpleCNN()
        elif model_name == "RegNetX":
            model = get_regnetx_model()
        elif model_name == "ComplexCNN":
            model = ComplexCNN()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Load data
        trainloader, testloader = load_and_visualize_data(batch_size)
        
        # Split training set for validation (or load a validation set if available)
        train_size = int(0.8 * len(trainloader.dataset))
        val_size = len(trainloader.dataset) - train_size
        val_subset = torch.utils.data.random_split(trainloader.dataset, [train_size, val_size])
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Define criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Train and evaluate
        train_model(model, trainloader, valloader, criterion, optimizer, epochs)
        accuracy = evaluate_model(model, testloader)
        
        # Save results
        results.append({**params, "accuracy": accuracy})
        
        # Update best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            print(f"New best accuracy: {best_accuracy:.2f}% with params {best_params}")
    
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Best Parameters: {best_params}")
    return best_params, best_accuracy, results

# 5. Plotting Hyperparameter Tuning Results
def plot_tuning_results(results):
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        plt.scatter(subset.index, subset["accuracy"], label=model)
    
    plt.title("Hyperparameter Tuning Results")
    plt.xlabel("Combination Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_losses(training_losses, validation_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss", marker="o")
    plt.plot(validation_losses, label="Validation Loss", marker="o")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

# Main Script Execution
if __name__ == "__main__":
    param_grid = {
        'model': ["SimpleCNN", "RegNetX"],
        'batch_size': [32, 64],
        'learning_rate': [0.001, 0.0005],
        'optimizer': ['adam', 'sgd'],
        'epochs': [10]
    }
    
    best_params, best_accuracy, results = hyperparameter_tuning(param_grid)
    print(f"Hyperparameter Tuning Complete. Best Accuracy: {best_accuracy:.2f}%")
    print(f"Best Parameters: {best_params}")
    
    # Plot results
    plot_tuning_results(results)
