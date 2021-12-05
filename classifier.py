# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

class NetMnist(nn.Module):
    def __init__(self, dataset_type):
        super(NetMnist, self).__init__()
        if dataset_type == 'mnist':
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
        elif dataset_type == 'dogs':
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(12544, 64) 
            self.fc2 = nn.Linear(64, 120)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Based On: https://github.com/zrsmithson/Stanford-dogs/blob/master/models.py
class NetDogs(nn.Module):

    def __init__(self, dataset_type):
        super(NetDogs, self).__init__()
        ## Define all the layers of this CNN, the only requirements are:
        ## This network takes in a square (224 x 224), RGB image as input
        ## 120 output channels/feature maps

        # 1 - input image channel (RGB), 32 output channels/feature maps, 4x4 square convolution kernel
        # 2x2 max pooling with 10% droupout
        # ConvOut: (32, 221, 221) <-- (W-F+2p)/s+1 = (224 - 4)/1 + 1
        # PoolOut: (32, 110, 110) <-- W/s
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p = 0.1)

        # 2 - 64 output channels/feature maps, 3x3 square convolution kernel
        # 2x2 max pooling with 20% droupout
        # ConvOut: (64, 108, 108) <-- (W-F+2p)/s+1 = (110 - 3)/1 + 1
        # PoolOut: (64, 54, 54) <-- W/s
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p = 0.2)

        # 3 - 128 output channels/feature maps, 2x2 square convolution kernel
        # 2x2 max pooling with 30% droupout
        # ConvOut: (128, 53, 53) <-- (W-F+2p)/s+1 = (54 - 2)/1 + 1
        # PoolOut: (128, 26, 26) <-- W/s
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p = 0.3)

        # 4 - 256 output channels/feature maps, 3x3 square convolution kernel
        # 2x2 max pooling with 30% droupout
        # ConvOut: (256, 24, 24) <-- (W-F+2p)/s+1 = (24 - 3)/1 + 1
        # PoolOut: (256, 12, 12) <-- W/s
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p = 0.4)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.dropout5 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout6 = nn.Dropout(p = 0.6)
        self.fc3 = nn.Linear(1000, 250)
        self.dropout7 = nn.Dropout(p = 0.7)
        self.fc4 = nn.Linear(250, 120)


    def forward(self, x):
        # convolutions
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.relu(self.bn4(self.conv4(x)))))

        #flatten
        x = x.view(x.size(0), -1)

        #fully connected
        x = self.dropout5(self.fc1(x))
        x = self.dropout6(self.fc2(x))
        x = self.dropout7(self.fc3(x))
        x = self.fc4(x)
        # a softmax layer to convert the 120 outputs into a distribution of class scores
        x = F.log_softmax(x, dim=1)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

class Classifier():
    def __init__(self, args, device, dataset_type):
        self.args = args
        self.device = device
        self.dataset_type = dataset_type

    def train_model(self, model, optimizer, train_loader, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if self.args.dry_run:
                    break

    def test_model(self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * acc))
        return acc

    def train(self, train_loader, test_loader, model_name):
        model = NetMnist(self.dataset_type) if self.dataset_type == "mnist" else NetDogs(self.dataset_type)
        model = model.to(self.device)
        optimizer = optim.Adadelta(model.parameters(), lr=self.args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            self.train_model(model, optimizer, train_loader, epoch)
            self.test_model(model, test_loader)
            scheduler.step()

        if self.args.save_model:
            torch.save(model.state_dict(), f"mnist_cnn_{model_name}.pt")
        return model
