import os
import torch
import torchvision
from torch.nn import functional as F
from networks import Net
from auxiliary import load_data, count_parameters

import torch.optim as optim

# Hyperparameters Convolutional Network
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
N = 1000

# Load the data
train_loader, test_loader = load_data(batch_size_train, batch_size_test)

# Build the network
network = Net()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# Set up folder for models
if not os.path.exists('./results'):
        os.makedirs('./results')

# Count and display number of parameters
count_parameters(network)

# Train the network
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            # Save the model
            torch.save(network.state_dict(), './results' + f'/{network.__class__.__name__}.ckpt')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# test the model before training
test()
# train and test it
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
