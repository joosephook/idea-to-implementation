import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F


def train(model, optimizer, train_loader, device):
    # Author: Ghassen Hamrouni
    # Modified for the Veriff Task by: Joosep Hook
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
    # Author: Ghassen Hamrouni
    # Modified for the Veriff Task by: Joosep Hook
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
