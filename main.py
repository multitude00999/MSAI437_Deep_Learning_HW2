# region Imports
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, BatchNorm2d, Dropout
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from statistics import mean
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
        transforms.ToTensor(),
        # normalize
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
        softmax = torch.exp(outputs).detach().cpu()
        prob = list(softmax.numpy())
        predictions.append(np.argmax(prob, axis=1))
        targets.append(y.cpu())


    training_accuracy = accuracy_score(targets, predictions)
    training_loss = np.average(training_loss)

    return training_loss, training_accuracy

def eval_model(cnn_model, criterion, valid_data):

    predictions = []
    target = []
    epoch_loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_data):
            x, y = x.to(device), y.to(device)

            output = cnn_model(x)
            loss = criterion(output, y)
            epoch_loss.append(loss.item())
            softmax = torch.exp(output).cpu()
            prob = list(softmax.numpy())
            prediction = np.argmax(prob, axis=1)
            predictions.append(prediction)
            target.append(y.cpu())

    accuracy = accuracy_score(target, predictions)
    epoch_loss = np.average(epoch_loss)

    return epoch_loss, accuracy, predictions, target

def run_CNN(train_data, valid_data):
    # model
    cnn_model = ConvolutionalNeuralNetwork().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    criterion = CrossEntropyLoss()

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

    plot_CNN_confusion_matrix(predictions, targets)


def plot_CNN_learning_curves(train_losses, valid_losses, train_accs, valid_accs):
    train_losses = [mean(train_loss) for train_loss in train_losses]
    train_accs = [mean(train_acc) for train_acc in train_accs]
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
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['REAL', 'FAKE'],
                yticklabels=['REAL', 'FAKE'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(classification_report(targets, preds, target_names=['Predicted Fake', 'Predicted True']))


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

