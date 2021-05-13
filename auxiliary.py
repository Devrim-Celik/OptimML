from prettytable import PrettyTable
import torch
import torchvision


def count_parameters(model):
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
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)


    trainset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
      batch_size=batch_size_test, shuffle=True)

    subset = list(range(0, len(trainset), 100))
    trainset_1 = torch.utils.data.Subset(trainset, subset)
    trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size_train,
                                                shuffle=True)
    return trainloader_1, test_loader
