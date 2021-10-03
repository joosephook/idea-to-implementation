import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F


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
        if device.type == "cuda":
            torch.use_deterministic_algorithms(False)
        loss.backward()
        ########################
        # CUDA reproducibility #
        ########################
        if device.type == "cuda":
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
