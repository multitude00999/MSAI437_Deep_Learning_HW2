# region Imports
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
# endregion Imports

# region Global Constants
IMG_WIDTH = 224  # width of image
IMG_HEIGHT = 224  # height of image
BATCH_SIZE = 64  # batch size
SEED = 76   # random seed
DROPOUT = 0.25   # dropout probability
LEARNING_RATE = 7e-3    # learning rate
NUM_EPOCHS = 30    # number of epochs
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
        shuffle=True
    )

    # define a loader for the validation data
    valid_data = torch.utils.data.DataLoader(
        valid_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return train_data, valid_data
# endregion Data Loading

# region CNN Class
class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(256),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(512),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=DROPOUT),
        )

        self.linear_layers = Sequential(
            Linear(512 * 14 * 14, 512),
            ReLU(inplace=True),
            Dropout(),
            Linear(512, 256),
            ReLU(inplace=True),
            Dropout(),
            Linear(256, 10),
            ReLU(inplace=True),
            Dropout(),
            Linear(10, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
# endregion CNN Class

def train_epoch(cnn_model, optimizer, criterion, train_data):
    train_loss = 0.0
    training_loss = []
    training_accuracy = []
    predictions = []
    targets = []
    for i, (x, y) in enumerate(train_data):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = cnn_model(x)
        loss = criterion(outputs, y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        softmax = torch.exp(outputs).detach().cpu()
        prob = list(softmax.numpy())
        predictions.append(np.argmax(prob, axis=1))
        targets.append(y)

    for i in range(len(predictions)):
        training_accuracy.append(accuracy_score(targets[i].cpu(), predictions[i]))

    training_loss = np.average(training_loss)
    training_accuracy = np.average(training_accuracy)*100

    return training_loss, training_accuracy

def eval_model(cnn_model, criterion, valid_data):

    predictions = []
    target = []
    accuracy = []
    epoch_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_data):
            x, y = x.to(device), y.to(device)

            output = cnn_model(x)
            loss = criterion(output, y)
            epoch_loss += loss.item()
            softmax = torch.exp(output).cpu()
            prob = list(softmax.numpy())
            prediction = np.argmax(prob, axis=1)
            predictions.append(prediction)
            target.append(y)

    for i in range(len(predictions)):
        accuracy.append(accuracy_score(target[i].cpu(), predictions[i]))

    return epoch_loss/(i+1), np.average(accuracy)*100

def run_CNN(train_data, valid_data):
    # model
    cnn_model = ConvolutionalNeuralNetwork().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    criterion = CrossEntropyLoss()

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(cnn_model, optimizer, criterion, train_data)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_loss, valid_acc = eval_model(cnn_model, criterion, valid_data)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        print('epoch:', epoch, '\t training loss:', train_loss, '\t training accuracy:', train_acc,
              '\t validation loss:', valid_loss, '\t validation accuracy:', valid_acc)

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

