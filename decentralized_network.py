import matplotlib.pyplot as plt
import numpy as np
import torch

import graph as g
from data import load_mnist_data
from node import Node


class DecentralizedNetwork:
    graphs = {"FullyConnectedGraph": g.FullyConnectedGraph,
              "BinomialGraph": g.BinomialGraph,
              "RingOfCliques": g.RingOfCliques,
              "CirculantGraph": g.CirculantGraph,
              "CycleGraph": g.CycleGraph,
              "Torus2D": g.Torus2D
              }

    optimizers = {"Adam": torch.optim.Adam}
    tasks = {"MNIST": load_mnist_data}

    def __init__(
            self,
            nr_nodes: int,
            nr_classes: int,
            allocation: str,
            graph_type: str,
            lr: float,
            training_epochs: int,
            optimizer_type: str,
            task_type: str,
            add_privacy: bool,
            epsilon: float,
            delta: float,
            subset: bool,
            batch_size: int,
            test_granularity=1,
    ):
        # Save the type of graph to be used
        self.graph_type = graph_type
        # Construct the graph
        self.graph = DecentralizedNetwork.graphs[graph_type](nr_nodes)
        # Save the number of nodes and classes to assign to each node in the graph
        self.nr_nodes = self.graph.n_nodes
        self.nr_classes = nr_classes
        self.allocation = allocation
        # Get the optimizer
        self.node_optimizer = [DecentralizedNetwork.optimizers[optimizer_type] for _ in range(self.nr_nodes)]
        # Save learning rate
        self.node_lr = lr

        # Set up differential privacy
        self.add_privacy = add_privacy
        self.training_epochs = training_epochs
        self.epsilon = epsilon
        self.delta = delta

        # Initialize the data loader
        self.task_type = task_type
        self.data_loader = DecentralizedNetwork.tasks[task_type]
        self.subset = subset
        self.batch_size = batch_size
        self.train_dataloader_list = None
        self.test_dataloader_list = None

        # Create lists for storing the test and training accuracies and losses
        self.test_accuracies_mean = []
        self.test_losses_mean = []
        self.test_accuracies_nodes = []
        self.test_losses_nodes = []
        self.sent_bits = []
        self.received_bits = []
        self.test_granularity = test_granularity
        self.epoch_list = []

        self.initialize_nodes()

    def initialize_nodes(self):
        # List for storing all agents
        self.nodes = []
        # Load the data
        self.train_dataloader_list, self.test_dataloader_list = self.data_loader(self.nr_nodes, self.nr_classes,
                                                                                 self.allocation, self.subset,
                                                                                 self.batch_size)
        for indx, (indx_, neighbours) in enumerate(self.graph.adj().items()):
            self.nodes.append(Node(
                str(indx),
                self.train_dataloader_list[indx],
                self.test_dataloader_list[indx],
                neighbours,
                self.node_optimizer[indx],
                self.node_lr,
                self.training_epochs,
                self.add_privacy,
                self.epsilon,
                self.delta
            ))
            print(f'Node_{indx} - Train Num: {len(self.train_dataloader_list[indx].dataset)} - Neighbors: {neighbours}')

    def train(self):
        for e in range(self.training_epochs):
            batch_iters = max([len(loader) for loader in self.train_dataloader_list])
            for batch in range(batch_iters):
                for n in self.nodes:
                    n(batch)

                # Share weights between the nodes
                self.share_weights()

            self.reset_sharing()

            # Print current performance
            if e % self.test_granularity == 0:
                self.store_performance(e)
                self.training_print(e)

    def reset_sharing(self):
        for a in self.nodes:
            a.iter_loader = iter(a.train_dataloader)
            a.stop_sharing = False

    def store_performance(self, e):
        performances = [a.test() for a in self.nodes]

        [a.save_train_per_epoch() for a in self.nodes]
        [a.reset_train_performance_() for a in self.nodes]

        self.test_accuracies_nodes.append([x[0] for x in performances])
        self.test_losses_nodes.append([x[1] for x in performances])

        self.test_accuracies_mean.append(np.mean(self.test_accuracies_nodes[-1]))
        self.test_losses_mean.append(np.mean(self.test_losses_nodes[-1]))

        self.sent_bits.append([node.sent_bytes for node in self.nodes])
        self.received_bits.append([node.received_bytes for node in self.nodes])

        self.epoch_list.append(e)

    def training_print(self, epoch):
        print(
            f"[{epoch:5d}] Validation Data: Accuracy {self.test_accuracies_mean[-1]:.3f} | Loss {self.test_losses_mean[-1]:.3f}")

    def share_weights(self):
        for sender_indx, sender in enumerate(self.nodes):
            for receiver_indx, receiver in enumerate(self.nodes):
                if receiver_indx in sender.neighbours:
                    a, b = sender.share_weights()
                    receiver.receive_weights(*sender.share_weights())

    def get_bytes(self):
        return [{f"sent_bytes_{n_indx + 1}": node.sent_bytes, f"received_bytes_{n_indx + 1}": node.received_bytes} for
                n_indx, node in enumerate(self.nodes)]

    def plot_training(self):
        plt.figure("MP")
        plt.title("Mean Performance")
        plt.plot(self.test_accuracies_mean, label="Test Accuracy")
        plt.plot(self.test_losses_mean, label="Test Losses")
        plt.legend()

        plt.figure("NP")
        plt.title("Node Performance")
        for n_indx in range(self.nr_nodes):
            plt.plot([x[n_indx] for x in self.test_accuracies_nodes], label=f"Node{n_indx}")
        plt.legend()

        plt.show()
