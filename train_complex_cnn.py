import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Dataset Loading and Visualization
def load_and_visualize_data(batch_size, valid_split=0.1):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
    # Compute mean and standard deviation
    mean = 0.0
    std = 0.0
    num_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten spatial dimensions
        mean += images.mean(2).sum(0)  # Sum mean per channel
        std += images.std(2).sum(0)    # Sum std per channel
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
    ])
    
    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Split training dataset into training and validation sets
    num_train = int((1 - valid_split) * len(dataset))
    num_valid = len(dataset) - num_train
    train_set, valid_set = torch.utils.data.random_split(dataset, [num_train, num_valid])
    
    # Create dataloaders
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Visualize a few images from the training set
    classes = dataset.classes
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    return trainloader, validloader, testloader, classes

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # Compute loss without reducing it

    def forward(self, x, target):
        n_class = x.size(1)
        # Compute cross entropy loss
        ce_loss = self.criterion(x, target)
        
        # Smooth the labels
        with torch.no_grad():
            target_one_hot = torch.zeros_like(x).to(x.device)  # Initialize a tensor of zeros (one-hot encoding)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)  # Set the target class index to 1
            target_one_hot = (1 - self.epsilon) * target_one_hot + self.epsilon / n_class  # Apply label smoothing
        
        # Compute the loss for each class
        smooth_loss = (-target_one_hot * F.log_softmax(x, dim=-1)).sum(dim=-1)
        
        return smooth_loss.mean()  # Return the mean loss across the batch


# Model Definitions
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
        x = x + residual  # Add residual connection
        x = self.pool3(x)

        # Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Training and Evaluation Functions
def train_model(model, trainloader, validloader, criterion, optimizer, epochs):
    model = model.to(device)  # Ensure model is on the correct device
    training_losses = []
    validation_losses = []
    training_acc = []
    validation_acc = []
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Now using the label smoothing loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1) 
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        training_loss = running_loss / len(trainloader)
        training_accuracy = (correct_train / total_train) * 100
        training_losses.append(training_loss)
        training_acc.append(training_accuracy)
        
        # Validation Phase
        model.eval()
        validation_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU/CPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)  
                validation_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        validation_loss = validation_loss / len(validloader)
        validation_accuracy = (correct_valid / total_valid) * 100
        validation_losses.append(validation_loss)
        validation_acc.append(validation_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%, "
              f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")
        # Update LR scheduler
        scheduler.step(validation_loss)

    print('Finished Training')
    return training_losses, validation_losses, training_acc, validation_acc


def test_model(model, testloader):
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
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Plot Results
def plot_results(name, train_losses, val_losses, train_accs, val_accs):
  # Plotting the loss and accuracy
  plt.figure(figsize=(12,5))

  plt.subplot(1, 2, 1)
  plt.plot(train_losses, label='Train Loss')
  plt.plot(val_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.title(f'Training and Validation Loss ({name})')

  plt.subplot(1, 2, 2)
  plt.plot(train_accs, label='Train Accuracy')
  plt.plot(val_accs, label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.legend()
  plt.title(f'Training and Validation Accuracy ({name})')

  plt.show()

# Main Script Execution
if __name__ == "__main__":
    learning_rate = 0.0005
    batch_size = 128
    num_epochs = 75
    name = "Adam"

    # Create the model, optimizer, and criterion
    final_model = ComplexCNN()
    trainloader, valloader, testloader, _ = load_and_visualize_data(batch_size=batch_size)
    criterion = LabelSmoothingCrossEntropy(epsilon=0.1)  # Use label smoothing with epsilon=0.1
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Train the model
    training_losses, validation_losses, training_acc, validation_acc = train_model(
        final_model, trainloader, valloader, criterion, optimizer, num_epochs
    )

    # Plot the results
    plot_results(name, training_losses, validation_losses, training_acc, validation_acc)

    # Test the model
    test_accuracy = test_model(final_model, testloader)


