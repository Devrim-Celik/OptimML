import torchvision
import torch
import numpy as np
import random
from agents import Agent
from graph import *


def create_agent_data(samples, labels, nr_agents):
    # Get random indices
    random_indices = sorted(np.random.randint(0, high=samples.shape[0], size=nr_agents - 1).tolist())
    random_indices = [0] + random_indices
    random_indices += [samples.shape[0]]
    # split the data for each agent: everyone gets a random amount of data
    agent_data = [samples[random_indices[i]:random_indices[i + 1]] for i in range(nr_agents)]
    agent_labels = [labels[random_indices[i]:random_indices[i + 1]] for i in range(nr_agents)]

    return agent_data, agent_labels


def load_MNIST_data(nr_agents):
    MNIST_TRAIN_DATA = torchvision.datasets.MNIST('./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]))
    MNIST_TEST_DATA = torchvision.datasets.MNIST('./data', train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))
                                                  ]))

    TR_SAMPLES = MNIST_TRAIN_DATA.data
    TR_LABELS = MNIST_TRAIN_DATA.targets
    TE_SAMPLES = MNIST_TEST_DATA.data
    TE_LABELS = MNIST_TEST_DATA.targets

    agent_train_data, agent_train_labels = create_agent_data(TR_SAMPLES, TR_LABELS, nr_agents)
    agent_test_data, agent_test_labels = create_agent_data(TE_SAMPLES, TE_LABELS, nr_agents)

    return agent_train_data, agent_test_data, agent_train_labels, agent_test_labels


graph = CycleGraph(10)
NR_AGENTS = graph.n_nodes
ALPHA = 0.9
LR = [0.01]*NR_AGENTS

# Load the MNIST data suited for number of agents
agent_train_data, agent_test_data, agent_train_labels, agent_test_labels = load_MNIST_data(NR_AGENTS)

optimizers = [torch.optim.Adam for _  in range(NR_AGENTS)]

# create the agents
agents = []
for i, nb in graph.adj().items():
    agents.append(Agent(str(i), agent_train_data[i],  agent_train_labels[i], agent_test_data[i],  agent_test_labels[i], nb, optimizers[i], LR[i], ALPHA))

epochs = 1000
for i in range(epochs):
    # update step
    [a() for a in agents]

    # share weights with neighbours
    for send_indx, sender_agent in enumerate(agents):
        for rec_indx, receiver_agent in enumerate(agents):
            if rec_indx in sender_agent.neighbours:
                receiver_agent.receive_weights(sender_agent._weights())

    # keep track of the loss and error of each agents: networks seem to learn
    # batch GD is implicitly implemented through the different agents
    loss_list = [a.training_loss() for a in agents]
    error_list = [a.nb_errors for a in agents]


    if i % 100 == 0:
        print(f"[+] Epoch: {i}, Training Loss: {np.mean(loss_list):.3f}, Training errors: {np.mean(error_list):.3f}")
        # test
        test_list = [a.test() for a in agents]
        test_error, test_loss = list(zip(*test_list))
        print(f"[-] Epoch: {i}, Test Loss: {np.mean(test_loss):.3f}, Test errors: {np.mean(test_error):.3f}")

