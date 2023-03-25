import os
import random
import numpy as np

import math

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch.nn as nn

from sklearn.metrics import accuracy_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

train_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=512, shuffle=True)

test_data = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=512, shuffle=False)

classes = train_data.classes

class Network(nn.Module):

    def __init__(self, num_classes=len(classes)):

        super(Network, self).__init__()

        self.conv = nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
          #  nn.ReLU(inplace=True),
          #  nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
          #  nn.ReLU(inplace=True),
          #  nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
          #  nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
          #  nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
          #  nn.ReLU(inplace=True),
          #  nn.MaxPool2d(kernel_size=2)
        ])

        self.classifier = nn.Sequential(*[
            nn.Linear(256 * 2 * 2, 4096),
            #nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        ]
                                        )

        self.masks = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(torch.ones([64, 3, 3, 3])), requires_grad=False),
             torch.nn.Parameter(torch.Tensor(torch.ones([192, 64, 3, 3])), requires_grad=False),
             torch.nn.Parameter(torch.Tensor(torch.ones([384, 192, 3, 3])), requires_grad=False),
             torch.nn.Parameter(torch.Tensor(torch.ones([256, 384, 3, 3])), requires_grad=False),
             torch.nn.Parameter(torch.Tensor(torch.ones([256, 256, 3, 3])), requires_grad=False),
             torch.nn.Parameter(torch.Tensor(torch.ones(4096, 256 * 2 * 2)), requires_grad=False),
             torch.nn.Parameter(torch.Tensor(torch.ones(4096, 4096)), requires_grad=False),
             torch.nn.Parameter(torch.Tensor(torch.ones(num_classes, 4096)), requires_grad=False)])

    def forward(self, x):

        # Multiplying the convolutional and linear layers with the mask

        with torch.no_grad():

            for i in range(len(self.conv)):
                self.conv[i].weight.copy_(self.conv[i].weight.data * self.masks[i])

            for i in range(len(self.classifier)):
                self.classifier[i].weight.copy_(self.classifier[i].weight.data * self.masks[i + 5])

        # Performing the forward pass

        # For convolution part

        for i in range(len(self.conv)):
            x = self.conv[i](x)
            x = nn.ReLU(inplace=True)(x)

            if i in [0, 1, 4]:

                x = nn.MaxPool2d(kernel_size=2)(x)

            else:
                pass


        # For fully connected part

        x = nn.Flatten()(x)

        for i in range(len(self.classifier)):

            if i < len(self.classifier) - 1:

                x = self.classifier[i](x)
                x = nn.ReLU(inplace=True)(x)

            else:

                return self.classifier[i](x)

        return x


def train_model(epochs, dataloader, device, model, optimizer, loss_function):
    # Set model to training mode in order to unfreeze all layers and allow gradient propagation
    model.train()

    # These two lists will be used to store average loss and accuracy for each epoch
    total_loss, acc = [], []

    # Now write out the training procedure
    for epoch in range(epochs):

        print("Epoch:", epoch + 1)

        # Each batch produces a loss, predictions and target
        batch_loss, batch_preds, batch_target = 0, [], []

        # For each batch, train the model
        for batch_idx, (x, y) in enumerate(dataloader):
            # Make sure that data is on the same device as the model
            x, y = x.to(device), y.to(device)

            # Remove all previous gradients
            optimizer.zero_grad()

            # Get predictions by performing a forward pass
            preds = model(x)

            # Calculate error
            loss = loss_function(preds, y)

            # Calculate all the gradients for each layer
            loss.backward()

            # Finall, update the weights
            optimizer.step()

            # Save the loss
            batch_loss += loss.item()

            # Save the predictions and target
            batch_preds.extend(np.argmax(preds.cpu().detach().numpy(), axis=1))
            batch_target.extend(y.cpu().detach().numpy())

        # Calculate average loss
        total_loss.append(batch_loss / len(dataloader))

        # Calculate accuracy for this epoch
        acc.append(accuracy_score(batch_target, batch_preds))
        print("Loss:", total_loss[-1], "\tAcc:", acc[-1])

    return model, total_loss, acc


def test_model(dataloader, device, model):
    # Set model to eval mode in order to freeze all layers so that no parameter gets updated during testing
    model.eval()

    # Each batch produces a loss, predictions and target
    batch_preds, batch_target = [], []

    # For each batch, train the model
    for batch_idx, (x, y) in enumerate(dataloader):

        # Make sure that data is on the same device as the model
        x, y = x.to(device), y.to(device)
        preds = model(x)

        # Save the predictions and target
        batch_preds.extend(np.argmax(preds.cpu().detach().numpy(), axis=1))
        batch_target.extend(y.cpu().detach().numpy())

    return accuracy_score(batch_target, batch_preds)


model = Network().to(device)

# Set the number of epochs to be used
epochs = 5

# Set the number of rounds
rounds = 5

# Set the sparsity level for each layer
sparsity = 0.1

# Create the model
model = Network().to(device)

# Define Loss
loss_function = torch.nn.CrossEntropyLoss()

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# First save the model weights that have been initialized
init_weights = [[model.conv[i].weight.data.to(device) for i in range(len(model.conv))],
                [model.classifier[i].weight.data.to(device) for i in range(len(model.classifier))]]

for round_ in range(rounds):

    print("\n\n\nROUND", round_ + 1, "Started\n----------------------")

    # First train the model for some epochs
    model, _, _ = train_model(epochs, train_loader,
                              device, model, optimizer, loss_function)
    if round_ == 0:

        print("Test Accuracy before pruning:",
              test_model(test_loader, device, model))

    else:
        print("Test Accuracy after pruning and retraining:",
              test_model(test_loader, device, model))

    with torch.no_grad():

        # Now prune the model weights
        for i in range(len(model.conv)):

            num_filters = model.conv[i].weight.shape[0]
            num_channels = model.conv[i].weight.shape[1]
            num_rows = model.conv[i].weight.shape[2]
            num_cols = model.conv[i].weight.shape[3]

            # Lottery Ticket Style Pruning
            indices = torch.argsort(torch.reshape(torch.abs(model.conv[i].weight.data),
                                                  (1, num_filters * num_channels * num_rows * num_cols)).squeeze())

            # Since we already have the indices to prune, first reset the parameters
            model.conv[i].weight.copy_(init_weights[0][i])

            # Now prune
            model.masks[i] = torch.reshape(model.masks[i],
                                           (1, num_filters * num_channels * num_rows * num_cols)).squeeze()

            val = ((sparsity * 100) ** ((round_ + 1) / rounds)) / 100

            model.masks[i][indices[:math.ceil(val * num_filters * num_channels * num_rows * num_cols)]] = 0

            model.masks[i] = torch.reshape(torch.reshape(
                model.masks[i], (1, num_filters * num_channels * num_rows * num_cols)).squeeze(),
                                           (num_filters, num_channels, num_rows, num_cols))

        for i in range(len(model.classifier)):

            n_rows = model.classifier[i].weight.data.shape[0]
            n_cols = model.classifier[i].weight.data.shape[1]

            # Lottery Ticket Style Pruning
            indices = torch.argsort(
                torch.reshape(torch.abs(model.classifier[i].weight.data), (1, n_rows * n_cols)).squeeze())

            # Since we already have the indices to prune, first reset the parameters
            model.classifier[i].weight.copy_(init_weights[1][i])

            # Now prune
            model.masks[i + 5] = torch.reshape(model.masks[i + 5], (1, n_rows * n_cols)).squeeze()
            val = ((sparsity * 100) ** ((round_ + 1) / rounds)) / 100

            model.masks[i + 5][indices[:math.ceil(val * n_rows * n_cols)]] = 0
            model.masks[i + 5] = torch.reshape(torch.reshape(model.masks[i + 5], (1, n_rows * n_cols)).squeeze(),
                                               (n_rows, n_cols))

    print("Test Accuracy after pruning but no retraining:", test_model(test_loader, device, model))
