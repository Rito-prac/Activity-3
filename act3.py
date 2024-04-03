pip install torchvision
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torchvision import datasets
from torchvision.models import resnet18



print("PyTorch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
# device = // Your code here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# transform = // Your code here
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ColorJitter(hue=0.5,saturation=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
# Filter only even-numbered classes
import torchvision.datasets

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Your code here
even_classes = [i for i in range(0,10,2)]

trainset_even = Subset(trainset,[i for i in range(len(trainset)) if trainset.targets[i] in even_classes])
testset_even = Subset(testset, [i for i in range(len(testset)) if testset.targets[i] in even_classes])
#HINT: Use torch utils Subset class to create a subset of the dataset
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
model = resnet18(weights='DEFAULT').to(device)
model.fc = nn.Linear(model.fc.in_features, len(even_classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train(model, trainloader, criterion, optimizer, device):

    model.train()

    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in trainloader:

        inputs, labels = inputs.to(device), labels.to(device)

        # Complete the code
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        train_loss+=loss.item()


        _, pred = outputs.max(1)
        total_train += labels.size(0)
        correct_train += pred.eq(labels).sum().item()

    train_acc = 100 * correct_train / total_train

    return train_loss/len(trainloader), train_acc
def test(model, testloader, criterion, device):
    model.eval()
    correct_test = 0
    total_test = 0

    # Your code here
    with torch.no_grad():
      for inputs,labels in testloader:
        inputs,labels = inputs.to(device),labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs,labels)

        _,pred = outputs.max(1)
        total_test +=labels.size(0)
        correct_test+=pred.eq(labels).sum().item()
    test_accuracy = 100 * correct_test / total_test

    return test_accuracy
def plot_accuracies(train_accuracies, test_accuracies, epochs):

    # Your code here
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracies vs Epochs')
    plt.legend()
    plt.show()
 import seaborn as sns   
def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()
    all_labels = []
    all_predictions = []

    num_classes = len(class_names)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    # Your code here
    with torch.no_grad():
      for inputs,labels in dataloader:
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        _,predictions = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    confusion_matrix = confusion_matrix(all_labels,all_predictions)


    plt.figure(figsize=(num_classes, num_classes))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
