import torchvision
import torch
import numpy as np
import random
from agents import Agent
from graph import *

graph = CycleGraph(10)
NR_AGENTS = graph.n_nodes
ALPHA = 0.9
LR = [0.01]*NR_AGENTS
MNIST_DATA = torchvision.datasets.MNIST('./data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                                ]))

#train_set, val_set = torch.utils.data.random_split(dataset, [8000, 2000])
TR_SAMPLES = MNIST_DATA.data
TR_LABELS = MNIST_DATA.targets

random_indices = sorted(np.random.randint(0, high=TR_SAMPLES.shape[0], size=NR_AGENTS-1).tolist())
random_indices = [0] + random_indices
random_indices += [TR_SAMPLES.shape[0]]

agent_data = [TR_SAMPLES[random_indices[i]:random_indices[i+1]] for i in range(NR_AGENTS)]
agent_labels = [TR_LABELS[random_indices[i]:random_indices[i+1]] for i in range(NR_AGENTS)]
# TODO  can have oneself as neighbour
# TODO can have same neighbour multiple times
optimizers = [torch.optim.Adam for _  in range(NR_AGENTS)]

# create the agents
agents = []
for i, nb in graph.adj().items():
    agents.append(Agent(str(i), agent_data[i],  agent_labels[i], nb, optimizers[i], LR[i], ALPHA))

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

# [[[receiver_agent.receive_weights(sender_agent._weights())] for rec_indx, receiver_agent in enumerate(agents) if rec_indx in sender_agent.neighbours] for send_indx, sender_agent in enumerate(agents)]
