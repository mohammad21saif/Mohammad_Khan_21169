import torch
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader, Subset
from torchvision import models
import numpy as np
import torch.optim as optim
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Transformation for Alexnet, VGG16 and Resnet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the SVHN dataset
#Change the root path
full_dataset = SVHN(root='/home/moonlab/dl_assign/data', split='train', transform=transform, download=True)
subset_indices = np.random.choice(len(full_dataset), len(full_dataset) // 4, replace=False)
dataset_subset = Subset(full_dataset, subset_indices)
# Split the dataset
train_size = int(0.8 * len(dataset_subset))
test_size = len(dataset_subset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset_subset, [train_size, test_size])
#data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Get the modified model
def get_modified_model(model_name):
    print(f"Loading {model_name}")
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif model_name in ['resnet18', 'resnet50', 'resnet101']:
        model = getattr(models, model_name)(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise ValueError("Unsupported model name")
    model = model.to(device)
    return model


# Training and evaluation
def train_and_evaluate(model, train_loader, test_loader):
    print('Training and Evaluating the model')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
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
    model_names = ['alexnet', 'vgg16', 'resnet18', 'resnet50', 'resnet101']
    for model_name in model_names:
        print('-------------------------------------')
        print(f"Training and evaluating {model_name}")
        model = get_modified_model(model_name)
        train_and_evaluate(model, train_loader, test_loader)
        # save_model(model, model_name)
        print('Finished Training and Evaluating the model')
        print('-------------------------------------')
# choose_model()
        