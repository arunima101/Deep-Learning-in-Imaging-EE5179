import torch
import torchvision
from torchvision.transforms import ToTensor
import numpy as np



def load_data():
    train_data=torchvision.datasets.MNIST(root='./traindata',train=True,download=True,transform=ToTensor())
    test_data=torchvision.datasets.MNIST(root='./testdata',train=False,download=True,transform=ToTensor())
    return np.array(train_data.data),convert_to_onehot(train_data.train_labels,10),np.array(test_data.data),convert_to_onehot(test_data.test_labels,10)

def convert_to_onehot(labels,num_classes):
    output = np.eye(num_classes)[np.array(labels).reshape(-1)]
    return output.reshape(list(np.shape(labels))+[num_classes])