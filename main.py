# region Imports
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.optim as optim
import pandas as pd
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
# endregion Imports

# region Global Constants
IMG_WIDTH = 224  # width of image
IMG_HEIGHT = 224  # height of image
BATCH_SIZE = 32  # batch size
SEED = 42   # random seed
DROPOUT = 0.3   # dropout probability
LEARNING_RATE = 5e-3    # learning rate
NUM_EPOCHS = 20    # number of epochs
CNN_MODEL_PATH = 'cnn_classification_model.pt'  # path for saved CNN model
NUM_CLASSES = 2     # number of class labels
# endregion Global Constants

# region Set-up & Configurations
# device set-up
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
# endregion Set-up & Configurations

# region Data Loading
def load_data(train_dir, valid_dir):
    # transformations
    transformation = transforms.Compose([
        # random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
        # random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # resize
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        # transform to tensors
        transforms.ToTensor()
    ])
    train_set = ImageFolder(
        root=train_dir, transform=transformation)
    valid_set = ImageFolder(
        root=valid_dir, transform=transformation)
    # define a loader for the training data
    train_data = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # define a loader for the validation data
    valid_data = torch.utils.data.DataLoader(
        valid_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_data, valid_data
# endregion Data Loading

# region CNN Class
class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
        )

        self.linear_layers = Sequential(
            Linear(128 * 14 * 14, 512),
            ReLU(inplace=True),
            Dropout(),
            Linear(512, 256),
            ReLU(inplace=True),
            Dropout(),
            Linear(256, 10),
            ReLU(inplace=True),
            Dropout(),
            Linear(10, 2)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
# endregion CNN Class

def run_CNN(train_data, valid_data):
    # model
    cnn_model = ConvolutionalNeuralNetwork().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    criterion = CrossEntropyLoss()

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = 0.0
        training_loss = []
        training_accuracy = []
        for i, (x, y) in enumerate(train_data):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = cnn_model(x)
            loss = criterion(outputs, y)

            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            num_correct = sum(torch.argmax(torch.sigmoid(outputs)) == y)
            acc = 100.0 * num_correct / len(y)
            training_accuracy.append(acc.item())

        training_loss = np.average(training_loss)
        avg_accuracy = np.average(training_accuracy)
        print('epoch: \t', epoch, '\t training loss: \t', training_loss, '\t training accuracy: \t', avg_accuracy)

    # save the model
    torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)


def run_AutoEncoder(train_data, valid_data):
    pass

# region Main Function
def main():

    # arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='CNN')
    params = parser.parse_args()
    torch.manual_seed(SEED)

    # data loading
    train_dir = "beans/train"
    valid_dir = "beans/valid"
    train_data, valid_data = load_data(train_dir, valid_dir)

    if params.model == 'CNN':
        run_CNN(train_data, valid_data)
    elif params.model == 'AutoEncoder':
        run_AutoEncoder(train_data, valid_data)
# endregion Main Function

if __name__ == '__main__':
    main()

