import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.data as data


transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms, download=True)

train_dataloader=data.DataLoader(train_dataset, batch_size=16, shuffle=True)

if __name__=='__main__':
    for imgs,labels in train_dataloader:
        print(imgs.size())
        print(labels.shape)
        break