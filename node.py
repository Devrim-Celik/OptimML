from networks import Net
from torch.nn import functional as F
import torch
import numpy as np
import random
from typing import List
import sys


old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info

class Node():
    def __init__(self,
                 name: str,
                 training_samples: torch.Tensor,
                 training_labels: torch.Tensor,
                 test_samples: torch.Tensor,
                 test_labels: torch.Tensor,
                 neighbours: List[str],
                 optimizer: torch.optim.Optimizer,
                 learning_rate: float,
                 alpha: float) -> None:
        self.name = name
        # get training samples and associated labels
        self.training_samples = training_samples
        self.training_labels = training_labels
        # get test samples and associated labels
        self.test_samples = test_samples
        self.test_labels = test_labels
        # get the list of neighbours, this node can communicate with
        self.neighbours = neighbours
        # initialize a list for storing the training losses
        self.test_losses = []
        self.test_accuracies = []
        # initialize the network
        self.network = Net()
        # save learning rate and the given optimizer
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.network.parameters(),
                                   lr=learning_rate)
        # save parameter of neighbour influence
        self.alpha = alpha
        # a list for storing the estimates of neighbours
        self.shared_weights = []
        self.loss = None
        self.output = None
        self.nb_errors = None

        # for saving send and received bytes
        self.sent_bytes = 0
        self.received_bytes = 0

    def __call__(self):

        # todo include shared estimates
        self.network.train()

        # shuffle samples randomly
        idx = torch.randperm(self.training_samples.shape[0])

        # only take a subset
        n = 100
        idx = idx[:n]
        samples = self.training_samples[idx]
        labels = self.training_labels[idx]

        # do one step of SGD, which includes iterating through all samples once
        for j in range(n):

            self.optimizer.zero_grad()

            # select random sample
            #r_indx = np.random.randint(0, self.training_samples.shape[0], 1)[0]
            sample, label = samples[j].unsqueeze(0).unsqueeze(0), labels[j]

            sample = sample.type(torch.FloatTensor)
            label = label.unsqueeze(0)
            self.output = self.network(sample)
            self.loss = F.nll_loss(self.output, label)
            self.loss.backward()
            self.optimizer.step()
            #self.nb_errors = self.calculate_proportion_errors(self.output, label)
            # manual update
            with torch.no_grad():

                if self.shared_weights:
                    weights_iter = zip(self.network.parameters(), *self.shared_weights)
                    for elements in weights_iter:
                        new_weight = sum([e / len(elements) for e in elements])
                        elements[0].copy_(new_weight)

            # after the calculations, we empty the shared estimates for the next step
            self.shared_weights = []


    def receive_weights(self, weights, byte_size):
        self.received_bytes += byte_size
        self.shared_weights.append(weights)

    def _weights(self):
        return self.network.parameters()

    def share_weights(self):
        shared_weights = self._weights()
        shared_weights_size = sys.getsizeof(shared_weights)
        self.sent_bytes += shared_weights_size
        return shared_weights, shared_weights_size

    def training_loss(self):
        return self.loss.data[0]

    def calculate_accuracy(self, output, label):
        # compute prediction and compare against training labels
        errors = 0
        for row in range(output.size(0)):
            if torch.argmax(output[row]) != label[row]:
                errors = errors + 1
        # return proportion of errors
        return (output.size(0) - errors) / output.size(0)

    def test(self):
        with torch.no_grad():
            # adjust shape of test samples
            test_samples = self.test_samples.unsqueeze(1).type(torch.FloatTensor)
            output = self.network(test_samples)
            test_labels = self.test_labels
            test_loss = F.nll_loss(output, test_labels).item()
            acc = self.calculate_accuracy(output, self.test_labels)

        return acc, test_loss
        # print(f"Test Loss: {test_loss:.4f}")
        # print(f"Test Accuracy: {(nr_correct / self.test_samples.shape[0]) * 100}%")
