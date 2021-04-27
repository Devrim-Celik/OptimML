from torch import nn
from torch.nn import functional as F
from prettytable import PrettyTable


NUM_TARGETS = 1
NUM_CLASSES = 10

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.__name__ = "ConvNet"
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn1 = nn.BatchNorm2d(16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn2 = nn.BatchNorm2d(32)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(2 * 2 * 64, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):

        out = self.bn1(self.layer1(x))
        out = self.bn2(self.layer2(out))
        out = self.layer3(out)
        
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    
    
    
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