import pickle

import torch
import torchvision
from prettytable import PrettyTable


def count_parameters(model):
    """
    Count number of parameters in the neural network

    :param model:               neural network
    :return:                    total number parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def load_data(batch_size_train, batch_size_test):
    """
    Data loading for network_test

    :param batch_size_train:    batch size of training data
    :param batch_size_test:     batch size of testing data
    :return:                    train dataloader, test dataloader
    """
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


def load_pickle(path):
    with open(f"{path}.pkl", 'rb') as f:
        data = pickle.load(f)
    return data
