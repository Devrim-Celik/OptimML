from networks import Net
from torch.nn import functional as F
import torchvision
import torch
import numpy as np
import random
from typing import List
class Agent():

    def __init__(self,
            training_samples: torch.Tensor,
            training_labels: torch.Tensor,
            neighbours: List[str],
            optimizer: torch.optim.Optimizer,
            learning_rate: float,
            alpha:float) -> None:
        # get training samples and associated labels
        self.training_samples = training_samples
        self.training_labels = training_labels
        # get the list of neighbous, this agend can communicate with
        self.neighbours = neighbours
        # intialize a list for storing the training losses
        self.train_losses = []
        # initialize the network
        self.network = Net()
        # save learning rate and the given optimizer
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.network.parameters(),
                                    lr = learning_rate)
        # save parameter of neighbour influence
        self.alpha = alpha
        # a list for storing the estimates of neighbours
        self.shared_weights = []



    def __call__(self):
        # todo include shared estimates


        self.network.train()
        self.optimizer.zero_grad()

        #select random sample
        r_indx = np.random.randint(0, self.training_samples.shape[0], 1)[0]
        print(r_indx)

        sample, label = self.training_samples[r_indx].unsqueeze(0).unsqueeze(0), self.training_labels[r_indx]
        sample = sample.type(torch.FloatTensor)
        label = label.unsqueeze(0)
        output = self.network(sample)
        loss = F.nll_loss(output, label)
        loss.backward()
        #self.optimizer.step()

        # manual update

        with torch.no_grad():
            if self.shared_weights:
                for weights, neighbour_weights in zip(self.network.parameters(), random.choice(self.shared_weights)):
                    new_weight = self.alpha * weights + (1 - self.alpha) * neighbour_weights - self.learning_rate * weights.grad
                    weights.copy_(new_weight)
            else:
                for weights in self.network.parameters():
                    new_weight = weights - self.learning_rate * weights.grad
                    weights.copy_(new_weight)
        print(output, label)
        print(len(self.shared_weights))
        # after the calculations, we empty the shared estimates for the
        # next step
        self.shared_weights = []

    def receive_weights(self, weights):
        self.shared_weights.append(weights)

    def _weights(self):
        return self.network.parameters()
