# region Imports
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import csv
import shutil
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import pickle
import itertools
import os
import skimage.transform as T
import skimage.io as io
import random
from skimage.util import random_noise
# endregion Imports

# region Global Constants
IMG_WIDTH = 224  # width of image
IMG_HEIGHT = 224  # height of image
BATCH_SIZE = 32  # batch size
SEED = 0  # random seed
DROPOUT = 0.5  # dropout probability
LEARNING_RATE = 0.001    # learning rate
NUM_EPOCHS = 10   # number of epochs
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

def define_transformation():
    # transformations
    transformation = transforms.Compose([
    # resize
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    # transform to tensors
    transforms.ToTensor(),
    # normalize
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformation
def load_data(train_dir, valid_dir):
    transformation = define_transformation()
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
        shuffle=False
    )
    return train_data, valid_data
# endregion Data Loading

# region CNN Class
class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 28 * 28)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# endregion CNN Class

def train_epoch(cnn_model, optimizer, criterion, train_data):
    training_loss = []
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
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(predicted.cpu())
        targets.append(y.cpu())

    accuracy = []
    for i in range(len(predictions)):
        accuracy.append(accuracy_score(targets[i], predictions[i]))
    training_loss = np.average(training_loss)
    training_accuracy = np.average(accuracy) * 100

    return training_loss, training_accuracy
def eval_model(cnn_model, criterion, valid_data):

    predictions = []
    targets = []
    epoch_loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_data):
            x, y = x.to(device), y.to(device)

            outputs = cnn_model(x)
            loss = criterion(outputs, y)
            epoch_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu())
            targets.append(y.cpu())

    accuracy = []
    for i in range(len(predictions)):
        accuracy.append(accuracy_score(targets[i], predictions[i]))
    valid_accuracy = np.average(accuracy) * 100
    epoch_loss = np.average(epoch_loss)

    return epoch_loss, valid_accuracy, predictions, targets
def run_CNN(train_data, valid_data):
    # model
    cnn_model = ConvolutionalNeuralNetwork().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    predictions = []
    targets = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(cnn_model, optimizer, criterion, train_data)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_loss, valid_acc, pred, target = eval_model(cnn_model, criterion, valid_data)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        predictions.append(pred)
        targets.append(target)

        print('epoch:', epoch, '\t training loss:', train_loss, '\t training accuracy:', train_acc,
              '\t validation loss:', valid_loss, '\t validation accuracy:', valid_acc)

    # save the model
    torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)

    # plots
    plot_CNN_learning_curves(train_losses, valid_losses, train_accs,
                             valid_accs)

    with open('CNN_preds.txt', 'wb') as f:
        pickle.dump(predictions, f)
        f.close()
    with open('CNN_targets.txt', 'wb') as f:
        pickle.dump(targets, f)
        f.close()
    predictions = list(itertools.chain.from_iterable(predictions))
    predictions = [item.numpy() for item in predictions]
    predictions = np.concatenate(predictions, axis=0)
    targets = list(itertools.chain.from_iterable(targets))
    targets = [item.numpy() for item in targets]
    targets = np.concatenate(targets, axis=0)
    plot_CNN_confusion_matrix(predictions[:-1], targets[:-1])

    # prediction on Blind set
    predict_blind()

def predict_blind():
    test_dir = 'beans/test/'
    cnn_model = ConvolutionalNeuralNetwork().to(device)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
    cnn_model.eval()
    test_set = ImageFolder(
        root=test_dir, transform=define_transformation())
    test_data = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    predictions = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_data):
            x, y = x.to(device), y.to(device)
            outputs = cnn_model(x)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu())
    predictions = [item.numpy() for item in predictions]
    predictions = np.concatenate(predictions, axis=0)
    output = []
    idx = 0
    for file_name in os.listdir(test_dir+'/blind'):
        label = 'Healthy' if predictions[idx] == 0 else 'Unhealthy'
        output.append({'filename':file_name, 'label':label})
        idx += 1
    myFile = open('CNN_blind_prediction.csv', 'w')
    writer = csv.writer(myFile)
    writer.writerow(['File Name', 'Label'])
    for dictionary in output:
        writer.writerow(dictionary.values())
    myFile.close()

def plot_CNN_learning_curves(train_losses, valid_losses, train_accs, valid_accs):
    epochs = [i for i in range(NUM_EPOCHS)]
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_accs, 'go-', label='Training Accuracy (CNN)')
    ax[0].plot(epochs, valid_accs, 'ro-', label='validation Accuracy (CNN)')
    ax[0].set_title('Training & Validation Accuracy (CNN)')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs, train_losses, 'go-', label='Training Loss (CNN)')
    ax[1].plot(epochs, valid_losses, 'ro-', label='Validation Loss (CNN)')
    ax[1].set_title('Training & Validation Loss (CNN)')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    plt.show()

def plot_CNN_confusion_matrix(preds, targets):
    cm = confusion_matrix(preds, targets)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['Healthy', 'Unhealthy'],
                yticklabels=['Healthy', 'Unhealthy'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(classification_report(targets, preds, target_names=['Predicted Healthy', 'Predicted Unhealthy']))


def run_AutoEncoder(train_data, valid_data):
    pass

# region Main Function
def main():

    # data loading
    train_dir = "beans/train"
    valid_dir = "beans/valid"

    '''
    Directory Structure - Already done, hence commenting the code
    '''
    # subfolder_names = ['healthy', 'unhealthy']
    # path_names = [train_dir, valid_dir]
    # # creating subfolders in the dataset to represent the two classes
    # for path_name in path_names:
    #     for subfolder_name in subfolder_names:
    #         subfolder_path = os.path.join(path_name, subfolder_name)
    #         os.makedirs(subfolder_path, exist_ok=True)
    #
    # # moving the data to their respective class folders
    # for path_name in path_names:
    #     for file_name in os.listdir(path_name):
    #         if (file_name.split('_')[0] == 'healthy'):
    #             file_path = os.path.join(path_name, file_name)
    #             subfolder_path = os.path.join(path_name, 'healthy')
    #             shutil.move(file_path, subfolder_path)
    #         else:
    #             file_path = os.path.join(path_name, file_name)
    #             subfolder_path = os.path.join(path_name, 'unhealthy')
    #             shutil.move(file_path, subfolder_path)

    '''
    Data Augmentation - Already done, hence commenting the function calls
    '''
    # healthy_augmented = 'beans/train/healthy/augmented/'
    # healthy = 'beans/train/healthy/'
    # unhealthy_augmented = 'beans/train/unhealthy/augmented/'
    # unhealthy = 'beans/train/unhealthy/'
    # # train - healthy
    # augment_data('beans/train/healthy/', 'beans/train/healthy/augmented/')
    # # train - unhealthy
    # augment_data('beans/train/unhealthy/', 'beans/train/unhealthy/augmented/')
    # for file in os.listdir(healthy_augmented):
    #     shutil.move(healthy_augmented + file, healthy)
    # for file in os.listdir(unhealthy_augmented):
    #     shutil.move(unhealthy_augmented + file, unhealthy)

    # arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='CNN')
    params = parser.parse_args()
    torch.manual_seed(SEED)

    train_data, valid_data = load_data(train_dir, valid_dir)

    if params.model == 'CNN':
        run_CNN(train_data, valid_data)
    elif params.model == 'AutoEncoder':
        run_AutoEncoder(train_data, valid_data)
# endregion Main Function

# region Data Augmentation
def augment_data(dir, savedir):
    for file in os.listdir(dir):
        if(file.endswith('.jpg') or file.endswith('.png')):
            image = io.imread(dir + file)
            operation = random.choice(['rotate', 'shift', 'fliplr', 'flipud', 'addnoise',
                                      'rotateAndAddNoise', 'shiftAndAddNoise'])
            if operation == 'rotate':
                rotate(image, file, savedir)
            elif operation == 'shift':
                shift(image, file, savedir)
            elif operation == 'fliplr':
                fliplr(image, file, savedir)
            elif operation == 'flipud':
                flipud(image, file, savedir)
            elif operation == 'addnoise':
                addNoise(image, file, savedir)
            elif operation == 'rotateAndAddNoise':
                rotateAndAddNoise(image, file, savedir)
            elif operation == 'shiftAndAddNoise':
                shiftAndAddNoise(image, file, savedir)

def rotate(image, file, savedir):
    rotated = T.rotate(image, 30, mode='wrap')
    tup = os.path.splitext(file)
    io.imsave(savedir+tup[0]+'_rotated'+tup[1], rotated)

def shift(image, file, savedir):
    transform = T.AffineTransform(translation=(25, 25))
    wrapShift = T.warp(image, transform, mode='wrap')
    tup = os.path.splitext(file)
    io.imsave(savedir+tup[0]+'_shifted'+tup[1], wrapShift)

def fliplr(image, file, savedir):
    flipLR = np.fliplr(image)
    tup = os.path.splitext(file)
    io.imsave(savedir+tup[0]+'_lrshifted'+tup[1], flipLR)

def flipud(image, file, savedir):
    flipIUD = np.flipud(image)
    tup = os.path.splitext(file)
    io.imsave(savedir+tup[0]+'_upshifted'+tup[1], flipIUD)

def addNoise(image, file, savedir):
    sigma = 0.155
    # add random noise to the image
    noisyRandom = random_noise(image, var=sigma ** 2)
    tup = os.path.splitext(file)
    io.imsave(savedir+tup[0]+'_addednoise'+tup[1], noisyRandom)

def rotateAndAddNoise(image, file, savedir):
    rotated = T.rotate(image, angle=30, mode='wrap')
    sigma = 0.155
    noisyRandom = random_noise(rotated, var=sigma ** 2)
    tup = os.path.splitext(file)
    io.imsave(savedir+tup[0]+'_rotatedAndNoise'+tup[1], noisyRandom)

def shiftAndAddNoise(image, file, savedir):
    transform = T.AffineTransform(translation=(25, 25))
    wrapShift = T.warp(image, transform, mode='wrap')
    sigma = 0.155
    noisyRandom = random_noise(wrapShift, var=sigma ** 2)
    tup = os.path.splitext(file)
    io.imsave(savedir+tup[0]+'shiftedAndNoise'+tup[1], noisyRandom)
# endregion Data Augmentation

if __name__ == '__main__':
    main()

