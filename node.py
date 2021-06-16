import sys
from typing import List

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils import module_modification
from torch.nn import functional as F

from networks import Net

old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


torch.Tensor.__repr__ = tensor_info


class Node:
    def __init__(self,
                 name: str,
                 train_dataloader,
                 test_dataloader,
                 neighbours: List[str],
                 optimizer: torch.optim.Optimizer,
                 learning_rate: float,
                 training_epochs: int,
                 add_privacy: bool,
                 epsilon: float,
                 delta: float) -> None:
        self.name = name
        self.train_dataloader = train_dataloader
        self.iter_loader = iter(self.train_dataloader)
        self.test_dataloader = test_dataloader
        # Get the list of neighbours; this node can communicate with them
        self.neighbours = neighbours
        # Initialize a list for storing the losses and accuracies
        self.test_losses = []
        self.test_accuracies = []
        self.train_losses_temp = []
        self.train_losses = []
        self.train_accuracies_temp = []
        self.train_accuracies = []
        # Initialize the network
        self.network = Net()
        # Save learning rate and the given optimizer
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.network.parameters(),
                                   lr=learning_rate)
        # A list for storing the weight estimates of neighbours
        self.shared_weights = []
        self.loss = None
        self.output = None
        self.nb_errors = None
        self.training_epochs = training_epochs
        self.target_epsilon = epsilon
        self.delta = delta
        self.add_privacy = add_privacy
        self.stop_sharing = False
        if self.add_privacy:
            # We need to change the batchnorm module to enable using the privacy engine
            self.network = module_modification.convert_batchnorm_modules(self.network)
            privacy_engine = PrivacyEngine(
                self.network,
                sample_rate=1 / len(self.train_dataloader.dataset),
                epochs=self.training_epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.delta,
                max_grad_norm=1.0,
            )
            privacy_engine.attach(self.optimizer)

        # For saving send and received bytes
        self.sent_bytes = 0
        self.received_bytes = 0

        # For NN: decide to use cpu or gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, batch):

        self.network.to(self.device)

        self.network.train()

        if batch < len(self.train_dataloader):
            sample, label = next(self.iter_loader)

            self.optimizer.zero_grad()

            sample = sample.unsqueeze(1).type(torch.FloatTensor)
            self.output = self.network(sample)
            self.loss = F.nll_loss(self.output, label)
            self.loss.backward()
            self.optimizer.step()

            self.train_losses_temp.append(self.loss.item())
            self.train_accuracies_temp.append(self.calculate_accuracy(self.output, label))
        else:
            self.stop_sharing = True

        with torch.no_grad():
            if self.shared_weights:
                weights_iter = zip(self.network.parameters(), *self.shared_weights)
                for elements in weights_iter:
                    new_weight = sum([e / len(elements) for e in elements])
                    elements[0].copy_(new_weight)

        # After the calculations, we empty the shared estimates for the next step
        self.shared_weights = []

    def receive_weights(self, weights, byte_size):
        self.received_bytes += byte_size
        if weights:
            self.shared_weights.append(weights)

    def _weights(self):
        return self.network.parameters()

    def share_weights(self):
        if not self.stop_sharing:
            shared_weights = self._weights()
            shared_weights_size = sys.getsizeof(shared_weights)
            self.sent_bytes += shared_weights_size
        else:
            shared_weights = []
            shared_weights_size = 0
        return shared_weights, shared_weights_size

    def training_loss(self):
        return self.loss.data[0]

    def calculate_accuracy(self, output, label):
        # Compute prediction and compare against training labels
        errors = 0
        for row in range(output.size(0)):
            if torch.argmax(output[row]) != label[row]:
                errors = errors + 1
        # Return proportion of errors
        return (output.size(0) - errors) / output.size(0)

    def reset_train_performance_(self):
        self.train_accuracies_temp = []
        self.train_losses_temp = []

    def save_train_per_epoch(self):
        self.train_accuracies.append(np.mean(self.train_accuracies_temp))
        self.train_losses.append(np.mean(self.train_losses_temp))

    def test(self):
        running_loss = 0
        accuracy = 0
        self.network.eval()
        with torch.no_grad():
            for samples, labels in self.test_dataloader:
                # adjust shape of test samples
                test_samples = samples.unsqueeze(1).type(torch.FloatTensor).to(self.device)
                output = self.network(test_samples)
                test_labels = labels.to(self.device)

                running_loss += F.nll_loss(output, test_labels).item()
                accuracy += self.calculate_accuracy(output, labels)

        return accuracy / len(self.test_dataloader), running_loss / len(self.test_dataloader)
