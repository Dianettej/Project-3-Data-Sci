import medmnist
from medmnist import INFO, Evaluator
from medmnist import DermaMNIST
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

validation_dataset = DermaMNIST(split="val", download=True)
training_dataset = DermaMNIST(split="train", download=True)
testing_dataset = DermaMNIST(split="test", download=True)

info = INFO["dermamnist"]
task = info['task']
n_channels = info['n_channels']
n_classes = 7

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

DataClass = getattr(medmnist, info['python_class'])

#size: 1003
print(validation_dataset)
#size: 7007
print(training_dataset)
#size: 2005
print(testing_dataset)

image = training_dataset.montage(length=20)
image.save("../data/figures/train_montage.png")

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

validation_dataset_torch = DataClass(split='val', transform=data_transform)
training_dataset_torch = DataClass(split='train', transform=data_transform)
testing_dataset_torch = DataClass(split='test', transform=data_transform)

train_loader = data.DataLoader(dataset=training_dataset_torch, shuffle=True)

train_loader = data.DataLoader(dataset=training_dataset_torch, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=testing_dataset, batch_size=2*BATCH_SIZE, shuffle=False)