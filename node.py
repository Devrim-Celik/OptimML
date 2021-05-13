from opacus import PrivacyEngine
from networks import Net
from torch.nn import functional as F
import torchvision
import torch
import numpy as np
import random
from typing import List
from opacus.utils import module_modification

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
                 alpha: float,
                 add_privacy: bool) -> None:
        self.name = name
        # get training samples and associated labels
        self.training_samples = training_samples
        self.training_labels = training_labels
        # get test samples and associated labels
        self.test_samples = test_samples
        self.test_labels = test_labels
        # get the list of neighbours, this node can communicate with
        self.neighbours = neighbours
        # intialize a list for storing the training losses
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
        self.add_privacy = add_privacy
        if self.add_privacy:
            self.network = module_modification.convert_batchnorm_modules(self.network)
            privacy_engine = PrivacyEngine(
                self.network,
                sample_rate=0.01,
                alphas=[10, 100],
                noise_multiplier=1.3,
                max_grad_norm=1.0,
            )
            privacy_engine.attach(self.optimizer)

    def __call__(self):

        # todo include shared estimates
        self.network.train()
        self.optimizer.zero_grad()

        # select random sample
        r_indx = np.random.randint(0, self.training_samples.shape[0], 1)[0]
        sample, label = self.training_samples[r_indx].unsqueeze(0).unsqueeze(0), self.training_labels[r_indx]

        sample = sample.type(torch.FloatTensor)
        label = label.unsqueeze(0)
        self.output = self.network(sample)
        self.loss = F.nll_loss(self.output, label)
        self.loss.backward()
        # self.optimizer.step()
        #self.nb_errors = self.calculate_proportion_errors(self.output, label)
        # manual update
        with torch.no_grad():
            if self.shared_weights:
                for weights, neighbour_weights in zip(self.network.parameters(), random.choice(self.shared_weights)):
                    new_weight = self.alpha * weights + (
                                1 - self.alpha) * neighbour_weights - self.learning_rate * weights.grad
                    weights.copy_(new_weight)
            else:
                for weights in self.network.parameters():
                    new_weight = weights - self.learning_rate * weights.grad
                    weights.copy_(new_weight)
        # after the calculations, we empty the shared estimates for the next step
        self.shared_weights = []

    def receive_weights(self, weights):
        self.shared_weights.append(weights)

    def _weights(self):
        return self.network.parameters()

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
            test_loss = F.nll_loss(output, test_labels)
            acc = self.calculate_accuracy(output, self.test_labels)

        return acc, test_loss
        # print(f"Test Loss: {test_loss:.4f}")
        # print(f"Test Accuracy: {(nr_correct / self.test_samples.shape[0]) * 100}%")
