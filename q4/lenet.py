import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transformation for LeNet-5
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])  
])


# Adjusted LeNet-5 for SVHN
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # Adjusted for 3-channel input
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*6*6)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the SVHN dataset
#Change the root path
train_dataset = datasets.SVHN(root='/home/moonlab/dl_assign/data', split='train', download=True, transform=transform)
#create subset
subset_indices = np.random.choice(len(train_dataset), len(train_dataset) // 4, replace=False)
dataset_subset = torch.utils.data.Subset(train_dataset, subset_indices) 
# Split the dataset
train_size = int(0.8 * len(dataset_subset))
test_size = len(dataset_subset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset_subset, [train_size, test_size])
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Training and evaluation
def train_and_evaluate(model, train_loader, test_loader):
    print('Training and Evaluating the model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct // total} %')


#Saving the model
# def save_model(model, model_name):
#     print(f'Saving the model {model_name}')
#     torch.save(model.state_dict(), f'./SavedModels/{model_name}.pt')


def choose_model():
    print('------------------------------------------')
    print('Training LeNet-5 on SVHN dataset')
    model = LeNet5().to(device)
    train_and_evaluate(model, train_loader, test_loader)
    # save_model(model, 'lenet_svhn')
    print('Finished Training and Evaluating the model')
    print('------------------------------------------')

# choose_model()