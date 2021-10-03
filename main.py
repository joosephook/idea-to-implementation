# -*- coding: utf-8 -*-
# License: BSD
# Author: Ghassen Hamrouni
# Modified for the Veriff Task by: Joosep Hook
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

import json

from itertools import product
from functools import partial
import pandas as pd
from six.moves import urllib

from models import Net, NetCoordConv, NetHomographic

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()


def train(model, optimizer, train_loader, device):
    # modified from:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        ########################
        # CUDA reproducibility #
        ########################
        if DEVICE.type == "cuda":
            torch.use_deterministic_algorithms(False)
        loss.backward()
        ########################
        # CUDA reproducibility #
        ########################
        if DEVICE.type == "cuda":
            torch.use_deterministic_algorithms(True)

        optimizer.step()


def test(model, test_loader, device):
    # modified from:

    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        targets = []
        preds = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            targets.append(target)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            preds.append(pred.view(-1))

            correct += pred.eq(target.view_as(pred)).sum().item()
        cm = confusion_matrix(torch.cat(targets).cpu(), torch.cat(preds).cpu())
        test_loss /= len(test_loader.dataset)
        return test_loss, correct / len(test_loader.dataset), cm


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

def reproducible_ops() -> torch.device:
    """
    Tells PyTorch to use deterministic algorithms for certain
    CUDA operations, where possible. Some non-deterministic operations
    will throw a RunTimeError when we tell PyTorch to use deterministic
    algorithms, so for these operations the determinism can be turned
    off on a case-by-case basis.

    :return: the device used for training
    """
    # Reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    return device


if __name__ == '__main__':
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    device = reproducible_ops()

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

    model_list = [
        ('CNN',       partial(Net, use_stn=False)),
        ('Baseline',  partial(Net, use_stn=True)),
        ('CoordConv', partial(NetCoordConv, use_stn=False)),
        ('CoordConvSTN', partial(NetCoordConv, use_stn=True)),
        ('NetHomographic', partial(NetHomographic, use_stn=True)),
    ]
    rows = []
    epochs = 1
    lr=0.01
    # lr scheduling
    gamma = 0.8
    step_size = 10
    for seed, (model_name, model_fn) in product(range(1), model_list):
        # same neural network weights
        torch.random.manual_seed(seed)

        model = model_fn().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        test_losses = []
        n_corrects = []
        cms = []

        for epoch in range(epochs):
            train(model, optimizer, train_loader, device)
            test_loss, n_correct, cm = test(model, test_loader, device)
            test_losses.append(test_loss)
            n_corrects.append(n_correct)
            print(test_loss, n_correct)
            cms.append(cm.tolist())
            scheduler.step()

        rows.append(dict(
            seed=seed,
            model_name=model_name,
            test_losses=test_losses,
            n_corrects=n_corrects,
            cms=cms,
        ))
        print(rows[-1])

    df = pd.DataFrame.from_records(rows)
    df.to_csv('results-3.csv')
