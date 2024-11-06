import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# Image transformations
t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_data = CIFAR10(root='./data', train=True, download=True, transform=t)
test_data = CIFAR10(root='./data', train=False, download=True, transform=t)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the CNN model
class cnnmodel(nn.Module):
    def __init__(self):
        super(cnnmodel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, criterion, and optimizer
model = cnnmodel()
criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
train_losses = []  # List to store training losses for each epoch
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for image, label in train_loader:
        output = model(image)
        loss = criterian(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)  # Store loss for this epoch
    print(f'Epoch-{epoch+1}/{num_epochs}, loss: {epoch_loss}')

# Evaluate the model
correct = 0
total = 0
running_loss = 0.0
test_losses = []   # List to store test losses for each epoch
model.eval()
with torch.no_grad():
    for img, label in test_loader:
        output = model(img)
        loss = criterian(output, label)
        _, predicted = torch.max(output, 1)
        
        # Count the correct predictions
        correct += (predicted == label).sum().item()
        total += label.size(0)
        running_loss += loss.item()

    test_loss = running_loss / len(test_loader)
    test_losses.append(test_loss)  # Store loss for testing
    accuracy = (correct / total) * 100
    print(f'Loss: {test_loss}')
    print(f'Accuracy: {accuracy}%')

# Plotting the loss
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()
