# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_

.. figure:: /_static/img/stn/FSeq.png

In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__

Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.

One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.
"""
# License: BSD
# Author: Ghassen Hamrouni
# Modified for the Veriff Task by: Joosep Hook

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from six.moves import urllib

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()


class CoordConv(torch.nn.Module):
    def __init__(self, with_r, x_dim, y_dim, in_channels, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.with_r = with_r
        self.xs = torch.tile(torch.linspace(-1, 1, x_dim).reshape(1, -1), (x_dim, 1)).reshape(1, 1, x_dim, y_dim).to(
            DEVICE)
        self.ys = torch.tile(torch.linspace(-1, 1, y_dim).reshape(-1, 1), (1, y_dim)).reshape(1, 1, x_dim, y_dim).to(
            DEVICE)
        in_channels = in_channels + 2
        if self.with_r:
            in_channels += 1
            self.rs = torch.sqrt(torch.square(self.xs) + torch.square(self.ys))
        self.conv = torch.nn.Conv2d(in_channels, *args, **kwargs).to(DEVICE)

    def forward(self, x):
        if self.with_r:
            augmented = torch.cat([x, self.xs.expand(x.size(0), -1, -1, -1), self.ys.expand(x.size(0), -1, -1, -1), self.rs.expand(x.size(0), -1, -1, -1) ], dim=1)
        else:
            augmented = torch.cat([x, self.xs.expand(x.size(0), -1, -1, -1), self.ys.expand(x.size(0), -1, -1, -1)], dim=1)
        return self.conv(augmented)


class AddCoords(nn.Module):
    def __init__(self, x_dim, y_dim, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r
        self.xs = torch.tile(torch.linspace(-1, 1, x_dim).reshape(1, -1), (x_dim, 1)).reshape(1, 1, x_dim, y_dim).to(
            DEVICE)
        self.ys = torch.tile(torch.linspace(-1, 1, y_dim).reshape(-1, 1), (1, y_dim)).reshape(1, 1, x_dim, y_dim).to(
            DEVICE)
        self.extra_channels = 2
        if self.with_r:
            self.extra_channels += 1
            self.rs = torch.sqrt(torch.square(self.xs) + torch.square(self.ys))

    def forward(self, x):
        if self.with_r:
            augmented = torch.cat([x, self.xs.expand(x.size(0), -1, -1, -1), self.ys.expand(x.size(0), -1, -1, -1), self.rs.expand(x.size(0), -1, -1, -1) ], dim=1)
        else:
            augmented = torch.cat([x, self.xs.expand(x.size(0), -1, -1, -1), self.ys.expand(x.size(0), -1, -1, -1)], dim=1)
        return augmented

    pass
class Net(nn.Module):
    ######################################################################
    # Depicting spatial transformer networks
    # --------------------------------------
    #
    # Spatial transformer networks boils down to three main components :
    #
    # -  The localization network is a regular CNN which regresses the
    #    transformation parameters. The transformation is never learned
    #    explicitly from this dataset, instead the network learns automatically
    #    the spatial transformations that enhances the global accuracy.
    # -  The grid generator generates a grid of coordinates in the input
    #    image corresponding to each pixel from the output image.
    # -  The sampler uses the parameters of the transformation and applies
    #    it to the input image.
    #
    # .. figure:: /_static/img/stn/stn-arch.png
    #
    # .. Note::
    #    We need the latest version of PyTorch that contains
    #    affine_grid and grid_sample modules.
    #
    def __init__(self, use_cuda=False, use_stn=True):
        super(Net, self).__init__()
        self.use_cuda = use_cuda
        self.use_stn = use_stn
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        # self.localization = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )

        # Spatial transformer localization-network
        self.loc_conv1 = nn.Conv2d(1, 8, kernel_size=7)
        self.loc_maxpool1 = nn.MaxPool2d(2, stride=2)
        self.loc_relu1 = nn.ReLU(True)
        self.loc_conv2 = nn.Conv2d(8, 10, kernel_size=5)
        self.loc_maxpool2 = nn.MaxPool2d(2, stride=2)
        self.loc_relu2 = nn.ReLU(True)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def localization(self, x):
        """
        torch.Size([64, 1, 28, 28])
        torch.Size([64, 8, 22, 22])
        torch.Size([64, 8, 11, 11])
        torch.Size([64, 10, 7, 7])
        torch.Size([64, 10, 3, 3])

        :param x:
        :return:
        """
        # print('localization')
        # print(x.size())
        x = self.loc_conv1(x)
        # print(x.size())
        x = self.loc_relu1(self.loc_maxpool1(x))
        # print(x.size())

        x = self.loc_conv2(x)
        # print(x.size())
        x = self.loc_relu2(self.loc_maxpool2(x))
        # print(x.size())
        # print()
        return x

    def stn(self, x):
        # Spatial transformer network forward function
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())

        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NetCoordConv(nn.Module):
    def __init__(self, use_coordconv=False, use_stn=False, use_r=False, use_cuda=True):
        super(NetCoordConv, self).__init__()
        self.use_cuda = use_cuda
        self.use_stn = use_stn
        self.use_coordconv = use_coordconv

        if self.use_coordconv:
            self.conv1 = CoordConv(use_r, 28, 28, 1, 10, kernel_size=5)
            self.conv2 = CoordConv(use_r, 12, 12, 10, 20, kernel_size=5)
        else:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        if self.use_coordconv:
            self.loc_conv1 = CoordConv(use_r, 28, 28, 1, 8, kernel_size=7)
        else:
            self.loc_conv1 = nn.Conv2d(1, 8, kernel_size=7)

        self.loc_maxpool1 = nn.MaxPool2d(2, stride=2)
        self.loc_relu1 = nn.ReLU(True)

        if self.use_coordconv:
            self.loc_conv2 = CoordConv(use_r, 11, 11, 8, 10, kernel_size=5)
        else:
            self.loc_conv2 = nn.Conv2d(8, 10, kernel_size=5)

        self.loc_maxpool2 = nn.MaxPool2d(2, stride=2)
        self.loc_relu2 = nn.ReLU(True)

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def localization(self, x):
        """
        torch.Size([64, 1, 28, 28])
        torch.Size([64, 8, 22, 22])
        torch.Size([64, 8, 11, 11])
        torch.Size([64, 10, 7, 7])
        torch.Size([64, 10, 3, 3])

        :param x:
        :return:
        """
        # print('localization')
        # print(x.size())
        x = self.loc_conv1(x)
        # print(x.size())
        x = self.loc_relu1(self.loc_maxpool1(x))
        # print(x.size())

        x = self.loc_conv2(x)
        # print(x.size())
        x = self.loc_relu2(self.loc_maxpool2(x))
        # print(x.size())
        # print()
        return x

    def stn(self, x):
        # Spatial transformer network forward function
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())

        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        if self.use_stn:
            x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, train_loader, device, epoch):
    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's use the SGD algorithm to train the model. The network is
    # learning the classification task in a supervised way. In the same time
    # the model is learning STN automatically in an end-to-end fashion.

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # reproducibility
        if model.use_cuda:
            torch.use_deterministic_algorithms(False)

        loss.backward()

        # reproducibility
        if model.use_cuda:
            torch.use_deterministic_algorithms(True)

        optimizer.step()
        # if batch_idx % 500 == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, device):
    #
    # A simple test procedure to measure the STN performances on MNIST.
    #
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        #       .format(test_loss, correct, len(test_loader.dataset),
        #               100. * correct / len(test_loader.dataset)))
        return test_loss




def convert_image_np(inp):
    ######################################################################
    # Visualizing the STN results
    # ---------------------------
    #
    # Now, we will inspect the results of our learned visual attention
    # mechanism.
    #
    # We define a small helper function in order to visualize the
    # transformations while training.
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(model, test_loader, device):
    # We want to visualize the output of the spatial transformers layer
    # after the training, we visualize a batch of input images and
    # the corresponding transformed batch using STN.
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


if __name__ == '__main__':
    # Reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        import os

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    print(f'Using device {device}')

    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=64, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)

    # with    stn: final test set average loss 0.0395 w/ lr=0.01
    # without stn: final test set average loss 0.0473 w/ lr=0.01
    # model = Net(use_cuda=torch.cuda.is_available(), use_stn=True).to(device)
    # model = Net(use_cuda=torch.cuda.is_available(), use_stn=False).to(device)
    # model = Net(use_cuda=torch.cuda.is_available(), use_stn=True).to(device)

    losses = {
        'with_stn': [],
        'without_stn': []
    }
    from itertools import product

    rows = []
    for seed, use_stn, use_coordconv, use_r in product(range(50), [True, False], [True, False], [True, False]):
        if not use_coordconv and use_r:
            # doesn't make sense
            continue

        torch.random.manual_seed(2)
        model =NetCoordConv(use_stn=use_stn, use_coordconv=use_coordconv, use_r=use_r).to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        test_losses = []

        for epoch in range(1, 50 + 1):
            train(model, optimizer, train_loader, device, epoch)
            test_loss = test(model, test_loader, device)
            test_losses.append(test_loss)

        rows.append(dict(
            seed=seed,
            use_stn=use_stn,
            use_coordconv=use_coordconv,
            use_r=use_r,
            test_losses=test_losses
        ))
        print(f'{use_stn=}\t{use_coordconv=}\t{use_r=}\tbest={np.min(test_losses):.4f}\t'+'\t'.join(f'{x:.4f}' for x in test_losses))

        # Visualize the STN transformation on some input batch
        # plt.ion()  # interactive mode
        # visualize_stn(model, test_loader, device)
        # plt.ioff()
        # plt.show()
    import pandas as pd
    df = pd.DataFrame.from_records(rows)
    df.to_csv('results.csv')
