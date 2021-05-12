import numpy as np
import torchvision


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
