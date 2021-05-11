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
    MNIST_DATA = torchvision.datasets.MNIST('./data', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]))

    train_set, test_set = torch.utils.data.random_split(MNIST_DATA, [8000, 2000])
    TR_SAMPLES = train_set.data
    TR_LABELS = train_set.dataset.targets
    TE_SAMPLES = test_set.dataset.data
    TE_LABELS = test_set.dataset.targets

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

epochs = 2
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
    print(f"Epoch: {i}, Loss: {np.mean(loss_list):.3f}, Nb_errors: {np.mean(error_list):.3f}")

    # test
    test_error, test_loss = agents[0].test()
    a
# [[[receiver_agent.receive_weights(sender_agent._weights())] for rec_indx, receiver_agent in enumerate(agents) if rec_indx in sender_agent.neighbours] for send_indx, sender_agent in enumerate(agents)]
