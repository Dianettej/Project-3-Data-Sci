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

validation_dataset = DermaMNIST(split="val", download=True)
training_dataset = DermaMNIST(split="train", download=True)
testing_dataset = DermaMNIST(split="test", download=True)

info = INFO["dermamnist"]
task = info['task']
n_channels = info['n_channels']
n_classes = 7

BATCH_SIZE = 128
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
    def forward(self, x):
        out = self.linear(x)
        return out

model = MulticlassLogisticRegression(2352, n_classes)

dataloader = DataLoader(training_dataset_transformed, batch_size=BATCH_SIZE, shuffle=True)

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