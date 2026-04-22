import medmnist
from medmnist import INFO, Evaluator
from medmnist import DermaMNIST
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from ignite.metrics import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

validation_dataset = DermaMNIST(split="val", download=True)
training_dataset = DermaMNIST(split="train", download=True)
testing_dataset = DermaMNIST(split="test", download=True)

info = INFO["dermamnist"]
task = info['task']
n_channels = info['n_channels']
n_classes = 7

BATCH_SIZE = 256
lr = 0.001

DataClass = getattr(medmnist, info['python_class'])

#size: 1003
print(validation_dataset)
#size: 7007
print(training_dataset)
#size: 2005
print(testing_dataset)

train_image = training_dataset.montage()
val_image = validation_dataset.montage()
test_image = testing_dataset.montage()

train_image.save("../data/figures/train_montage.png")
val_image.save("../data/figures/val_montage.png")
test_image.save("../data/figures/test_montage.png")

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

val_tensor = data_transform(val_image)
train_tensor = data_transform(train_image)
test_tensor = data_transform(test_image)

val_tensor_flat = val_tensor.flatten(start_dim = 1)
train_tensor_flat = train_tensor.flatten(start_dim = 1)
test_tensor_flat = test_tensor.flatten(start_dim = 1)

validation_dataset_transformed = DermaMNIST(split="val", download=True, transform=data_transform)
training_dataset_transformed = DermaMNIST(split="train", download=True, transform=data_transform)
testing_dataset_transformed = DermaMNIST(split="test", download=True, transform=data_transform)

class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.bn = nn.BatchNorm1d(input_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = MulticlassLogisticRegression(2352, n_classes)

dataloader = DataLoader(training_dataset_transformed, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(validation_dataset_transformed, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(testing_dataset_transformed, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Flatten inputs for linear model
        inputs = inputs.flatten(start_dim=1)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        
 
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        inputs = inputs.flatten(start_dim=1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()
    accuracy = correct / total
    print('Accuracy:', accuracy)

cm = confusion_matrix(testing_dataset_transformed, predictions)

"""class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2352, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork()

train_images, train_labels = next(iter(dataloader))
logits = model(train_images)

pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")"""