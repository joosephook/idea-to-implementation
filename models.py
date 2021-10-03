import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# from spatial_transformer_tutorial import DEVICE


class CoordConv(torch.nn.Module):
    def __init__(self, with_r, x_dim, y_dim, in_channels, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.with_r = with_r
        x_coords = torch.tile(torch.linspace(-1, 1, x_dim).reshape(1, -1), (x_dim, 1)).reshape(1, 1, x_dim, y_dim)
        y_coords = torch.tile(torch.linspace(-1, 1, y_dim).reshape(-1, 1), (1, y_dim)).reshape(1, 1, x_dim, y_dim)
        self.register_buffer('xs', x_coords)
        self.register_buffer('ys', y_coords)

        in_channels = in_channels + 2
        if self.with_r:
            in_channels += 1
            self.register_buffer('rs', torch.sqrt(torch.square(self.xs) + torch.square(self.ys)))
            
        self.conv = torch.nn.Conv2d(in_channels, *args, **kwargs)

    def forward(self, x):
        if self.with_r:
            augmented = torch.cat([x, self.xs.expand(x.size(0), -1, -1, -1), self.ys.expand(x.size(0), -1, -1, -1), self.rs.expand(x.size(0), -1, -1, -1) ], dim=1)
        else:
            augmented = torch.cat([x, self.xs.expand(x.size(0), -1, -1, -1), self.ys.expand(x.size(0), -1, -1, -1)], dim=1)
        return self.conv(augmented)


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
    def __init__(self, use_stn=True, **kwargs):
        super(Net, self).__init__()
        self.use_stn = use_stn
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        # Spatial transformer network forward function
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
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
    def __init__(self, use_stn=False, **kwargs):
        super(NetCoordConv, self).__init__()
        self.use_stn = use_stn

        self.conv1 = CoordConv(False, 28, 28, 1, 10, kernel_size=5)
        self.conv2 = CoordConv(False, 12, 12, 10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network

        if self.use_stn:
            self.localization = nn.Sequential(
                CoordConv(False, 28, 28, 1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                CoordConv(False, 11, 11, 8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 3 * 3, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def stn(self, x):
        # Spatial transformer network forward function
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)

        x = F.grid_sample(x, grid, align_corners=False)
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


class NetHomographic(nn.Module):
    def __init__(self, **kwargs):
        super(NetHomographic, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            CoordConv(False, 28, 28, 1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            CoordConv(False, 11, 11, 8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 8 parameter homographic transform
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 8)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))

    def transformImage(self, image, p):
        # modified from: https://github.com/chenhsuanlin/inverse-compositional-STN/blob/master/MNIST-pytorch/warp.py
        batchSize, channels, H, W = image.size()
        pMtrx = self.vec2mtrx(batchSize, p)

        refMtrx = torch.eye(3).cuda()
        refMtrx = refMtrx.repeat(image.size(0), 1, 1)
        transMtrx = refMtrx.matmul(pMtrx)

        # warp the canonical coordinates / generate the grid
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        X, Y = X.flatten(), Y.flatten()
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
        XYhom = np.tile(XYhom, [batchSize, 1, 1]).astype(np.float32)
        XYhom = torch.from_numpy(XYhom).cuda()
        XYwarpHom = transMtrx.matmul(XYhom)
        XwarpHom, YwarpHom, ZwarpHom = torch.unbind(XYwarpHom, dim=1)
        Xwarp = (XwarpHom / (ZwarpHom + 1e-8)).reshape(batchSize, H, W)
        Ywarp = (YwarpHom / (ZwarpHom + 1e-8)).reshape(batchSize, H, W)
        grid = torch.stack([Xwarp, Ywarp], dim=-1)

        return F.grid_sample(image, grid, align_corners=False)

    def vec2mtrx(self, batchSize, p):
        # modified from: https://github.com/chenhsuanlin/inverse-compositional-STN/blob/master/MNIST-pytorch/warp.py
        # homographic transformation from vec2mtrx
        I = torch.ones(batchSize, dtype=torch.float32).cuda()
        O = torch.zeros(batchSize, dtype=torch.float32).cuda()
        p1, p2, p3, p4, p5, p6, p7, p8 = torch.unbind(p, dim=1)
        pMtrx = torch.stack([torch.stack([O + p1, p2, p3], dim=-1),
                             torch.stack([p4, O + p5, p6], dim=-1),
                             torch.stack([p7, p8, I], dim=-1)], dim=1)
        return pMtrx


    def stn(self, x):
        # Spatial transformer network forward function
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 8)

        return self.transformImage(x, theta)

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)