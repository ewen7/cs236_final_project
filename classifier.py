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
import pdb

class Net(nn.Module):
    def __init__(self, dataset_type):
        super(Net, self).__init__()
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

    def forward(self, x): # dogs: [1000, 3, 32, 32]
        # pdb.set_trace()
        x = self.conv1(x) # dogs: [1000, 32, 30, 30]
        x = F.relu(x)
        x = self.conv2(x) # dogs: [1000, 64, 28, 28]
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # dogs: [1000, 64, 14, 14]
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # dogs: [1000, 12544]
        x = self.fc1(x) # dogs: [1000, 64]
        x = F.relu(x)
        x = self.dropout2(x) 
        x = self.fc2(x) # dogs: [1000, 120]
        output = F.log_softmax(x, dim=1) # dogs: [1000, 120]
        return output

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
        print("test length", len(test_loader))
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pdb.set_trace()
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                print("batch, ", test_loss, correct)

        test_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)
        print("total", test_loss, acc)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * acc))
        return acc

    def train(self, train_loader, test_loader, model_name):
        model = Net(self.dataset_type) #if self.dataset_type == "mnist" else NetDogs(self.dataset_type)
        model = model.to(self.device)
        optimizer = optim.Adadelta(model.parameters(), lr=self.args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            self.train_model(model, optimizer, train_loader, epoch)
            #self.test_model(model, test_loader)
            scheduler.step()

        if self.args.save_model:
            torch.save(model.state_dict(), f"mnist_cnn_{model_name}.pt")
        return model
