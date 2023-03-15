# region Imports
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
# endregion Imports

# region Global Constants
IMG_WIDTH = 128  # width of image
IMG_HEIGHT = 128  # height of image
BATCH_SIZE = 32  # batch size
SEED = 42 # random seed
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
        # # random horizontal flip
        # transforms.RandomHorizontalFlip(0.5),
        # # random vertical flip
        # transforms.RandomVerticalFlip(0.3),
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
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),


            nn.Flatten(),
            nn.Linear(262144, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, xb):
        return self.network(xb)
# endregion CNN Class

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, optimizer):
    history = []
    for epoch in range(epochs):

        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history
# region CNN training
def train_epoch(model, opt, criterion, dataloader):
  model.train()
  losses = []
  accs = []
  for i, (x, y) in enumerate(dataloader):
      x, y = x.to(device), y.to(device)
      opt.zero_grad()
      # forward pass
      pred =torch.sigmoid(model(x))
      print('pred', pred)
      print('y', y)
      # loss
      loss = criterion(pred, y)
      # backward pass
      loss.backward()
      # update weights
      opt.step()
      losses.append(loss.item())
      # accuracy
      num_correct = sum(pred == y)
      acc = 100.0 * num_correct/len(y)
      accs.append(acc.item())
      if (i%20 == 0):
          print("Batch " + str(i) + " : training loss = " + str(loss.item()) + "; training acc = " + str(acc.item()))
  return losses, accs
# endregion CNN training

# region LSTM Evaluation
def eval_model(model, criterion, evalloader):
  model.eval()
  total_epoch_loss = 0
  total_epoch_acc = 0
  preds = []
  with torch.no_grad():
      for i, (x, y) in enumerate(evalloader):
          x, y = x.to(device), y.to(device).to(torch.float32)
          pred = torch.argmax(model(x), 1).to(torch.float32)
          loss = criterion(pred, y)
          num_correct = sum(pred == y)
          acc = 100.0 * num_correct/len(y)
          total_epoch_loss += loss.item()
          total_epoch_acc += acc.item()
          preds.append(pred)

  return total_epoch_loss/(i+1), total_epoch_acc/(i+1), preds
def evaluate(model, opt, criterion, training_dataloader, valid_dataloader):
  train_losses = []
  valid_losses = []
  train_accs = []
  valid_accs = []

  print("Start Training...")
  for e in range(NUM_EPOCHS):
      print("Epoch " + str(e+1) + ":")
      losses, acc = train_epoch(model, opt, criterion, training_dataloader)
      train_losses.append(losses)
      train_accs.append(acc)
      valid_loss, valid_acc, val_preds = eval_model(model, criterion, valid_dataloader)
      valid_losses.append(valid_loss)
      valid_accs.append(valid_acc)
      print("Epoch " + str(e+1) + " : Validation loss = " + str(valid_loss) + "; Validation acc = " + str(valid_acc))
  return train_losses, valid_losses, train_accs, valid_accs
# endregion LSTM Evaluation
def run_CNN(train_data, valid_data):
    # model
    cnn_model = ConvolutionalNeuralNetwork().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # train_losses, valid_losses, train_accs, valid_accs = \
    #     evaluate(cnn_model, optimizer, criterion, train_data, valid_data)
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):

        # Evaluation and training on training dataset
        cnn_model.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_data):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = cnn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_accuracy += int(torch.sum(prediction == labels.data))

        train_accuracy = train_accuracy / len(train_data)
        train_loss = train_loss / len(train_data)

        # Evaluation on testing dataset
        cnn_model.eval()

        test_accuracy = 0.0
        for i, (images, labels) in enumerate(valid_data):
            images, labels = images.to(device), labels.to(device)

            outputs = cnn_model(images)
            _, prediction = torch.max(outputs.data, 1)
            test_accuracy += int(torch.sum(prediction == labels.data))

        test_accuracy = test_accuracy / len(valid_data)

        print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
            train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

        # Save the best model
        if test_accuracy > best_accuracy:
            torch.save(cnn_model.state_dict(), 'best_checkpoint.model')
            best_accuracy = test_accuracy

    # save the model
    #torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)


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

