import graph as g
import torch
from data import load_mnist_data
from node import Node
import numpy as np
import matplotlib.pyplot as plt

class DecentralizedNetwork():
    graphs = {"FullyConnectedGraph": g.FullyConnectedGraph,
              "BinomialGraph": g.BinomialGraph,
              "RingOfCliques": g.RingOfCliques,
              "CirculantGraph": g.CirculantGraph,
              "CycleGraph": g.CycleGraph
    }

    optmizers = {"Adam": torch.optim.Adam}

    tasks = {"MNIST": load_mnist_data}

    def __init__(
        self,
        nr_nodes: int,
        nr_classes: int,
        allocation: str,
        graph_type: str,
        alpha: float,
        lr: float,
        training_epochs: int,
        optimizer_type: str, # not used
        task_type: str, # not used
    ):
        # save the type of graph to be used
        self.graph_type = graph_type
        # construct the graph
        # TODO what happens if graph takes more than 1 argument?
        self.graph = DecentralizedNetwork.graphs[graph_type](nr_nodes)
        # save the number of nodes and classes to assign to each node in the graph
        self.nr_nodes = nr_nodes
        self.nr_classes = nr_classes
        self.allocation = allocation
        # get the optimizer
        self.node_optimizer = [DecentralizedNetwork.optmizers[optimizer_type] for _ in range(nr_nodes)]
        # save parameters
        self.node_lr = lr
        self.node_alpha = alpha
        self.training_epochs = training_epochs

        # initialize data loader
        self.task_type = task_type
        self.data_loader = DecentralizedNetwork.tasks[task_type]

        # for storing the test and training accuraries/loss
        self.test_accuracies_mean = []
        self.test_losses_mean = []
        self.test_accuracies_nodes = []
        self.test_losses_nodes = []
        # intialize nodes
        self.initialize_nodes()

    def initialize_nodes(self):
        # list for storing all agents
        self.nodes = []
        # load the data
        node_tr_data, node_te_data, node_tr_labels, node_te_labels = self.data_loader(self.nr_nodes, self.nr_classes,
                                                                                       self.allocation)

        for indx, neighbours in self.graph.adj().items():
             self.nodes.append(Node(
                str(indx),
                node_tr_data[indx],
                node_tr_labels[indx],
                node_te_data[indx],
                node_te_labels[indx],
                neighbours,
                self.node_optimizer[indx],
                self.node_lr,
                self.node_alpha
            ))

    def train(self):
        for e in range(self.training_epochs):
            # do one SGD step for all nodes
            for n in self.nodes:
                n()

            # share weights between the nodes
            self.share_weights()

            # calculate current test performance and store them
            #self.store_performance_test()

            # print current performance
            if e % 100 == 0:
                self.store_performance_test()
                self.training_print(e)

    def store_performance_test(self):
        performances = [a.test() for a in self.nodes]

        self.test_accuracies_nodes.append([x[0] for x in performances])
        self.test_losses_nodes.append([x[1] for x in performances])

        self.test_accuracies_mean.append(np.mean(self.test_accuracies_nodes[-1]))
        self.test_losses_mean.append(np.mean(self.test_losses_nodes[-1]))

    def training_print(self, epoch):
        print(f"[{epoch:5d}] Validation Data: Accuracy {self.test_accuracies_mean[-1]:.3f} | Loss {self.test_losses_mean[-1]:.3f}")


    def share_weights(self):
        for sender_indx, sender in enumerate(self.nodes):
            for receiver_indx, receiver in enumerate(self.nodes):
                if receiver_indx in sender.neighbours:
                    receiver.receive_weights(*sender.share_weights())

    def get_bytes(self):
        return [{f"sent_bytes_{n_indx+1}":node.sent_bytes, f"received_bytes_{n_indx+1}":node.received_bytes} for n_indx, node in enumerate(self.nodes)]

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
