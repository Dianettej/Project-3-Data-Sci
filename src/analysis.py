from medmnist import DermaMNIST
from PIL import Image

validation_dataset = DermaMNIST(split="val", download=True)
training_dataset = DermaMNIST(split="train", download=True)
testing_dataset = DermaMNIST(split="test", download=True)

#size: 1003
print(validation_dataset)
#size: 7007
print(training_dataset)
#size: 2005
print(testing_dataset)

image = training_dataset.montage(length=20)
image.save("../data/figures/train_montage.png")