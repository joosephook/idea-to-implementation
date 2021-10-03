# -*- coding: utf-8 -*-
# License: BSD
# Author: Ghassen Hamrouni
# Modified for the Veriff Task by: Joosep Hook
from __future__ import print_function

from functools import partial
from itertools import product

import pandas as pd
import torch
import torch.optim as optim
from six.moves import urllib
from torchvision import datasets, transforms

from models import Net, NetCoordConv, NetHomographic
from util import train, test, reproducible_ops

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
        ('CNN', partial(Net, use_stn=False)),
        ('Baseline', partial(Net, use_stn=True)),
        ('CoordConv', partial(NetCoordConv, use_stn=False)),
        ('CoordConvSTN', partial(NetCoordConv, use_stn=True)),
        ('NetHomographic', partial(NetHomographic, use_stn=True)),
        ('NetHomographic4', partial(NetHomographic, use_stn=True, iterations=4)),
    ]
    rows = []
    epochs = 50
    lr = 0.01
    # lr scheduling
    gamma = 0.8
    step_size = 10
    n_seeds = 30
    for seed, (model_name, model_fn) in product(range(n_seeds), model_list):
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
    df.to_csv('results.csv')
