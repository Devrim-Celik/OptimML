import torchvision
import torch
import numpy as np
import random
from agents import Agent

NR_NEIGHBOURS = 3
NR_AGENTS = 10
ALPHA = 0.9
LR = [0.01]*NR_AGENTS
MNIST_DATA = torchvision.datasets.MNIST('./data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                                ]))

TR_SAMPLES = MNIST_DATA.data
TR_LABELS = MNIST_DATA.targets

random_indices = sorted(np.random.randint(0, high=TR_SAMPLES.shape[0], size=NR_AGENTS-1).tolist())
random_indices = [0] + random_indices
random_indices += [TR_SAMPLES.shape[0]]

agent_data = [TR_SAMPLES[random_indices[i]:random_indices[i+1]] for i in range(NR_AGENTS)]
agent_labels = [TR_LABELS[random_indices[i]:random_indices[i+1]] for i in range(NR_AGENTS)]
# TODO  can have oneself as neighbour
# TODO can have same neighbour multiple times
neighbours = [[random.choice(list(range(NR_AGENTS))) for _ in range(NR_NEIGHBOURS)] for _ in range(NR_AGENTS)]
optimizers = [torch.optim.Adam for _  in range(NR_AGENTS)]

agents = [Agent(agent_data[i], agent_labels[i], neighbours[i], optimizers[i], LR[i], ALPHA) for i in range(NR_AGENTS)]
for i in range(10000):
    # update step
    [a() for a in agents]
    # sare weights with ALL other neighbours TODO only with neighbours

    [[[receiver_agent.receive_weights(sender_agent._weights())] for rec_indx, receiver_agent in enumerate(agents) if rec_indx in sender_agent.neighbours] for send_indx, sender_agent in enumerate(agents)]
