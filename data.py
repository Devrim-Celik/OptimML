import numpy as np
import torch
import torchvision
from matplotlib import pyplot
from torch.utils.data import DataLoader, Dataset


class LoadData:
    def __init__(self,
                 dataset: str,
                 train: bool,
                 subset: bool):
        """
        Class handles loading data for many separate nodes.

        :param dataset:             Which dataset to load
        :param train:               boolean to load train or test dataset
        :param subset:              Only load a fraction of data. Hardcoded to 30%.
        """

        PERCENT = .3

        if dataset == 'MNIST':
            data = torchvision.datasets.MNIST('./data', train=train, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ]))
        else:
            raise ValueError

        data_size = len(data)
        self.data = data

        if subset:
            indx = torch.randperm(data_size)[:int(data_size * PERCENT)]
            self.samples = self.data.data[indx, :, :]
            self.labels = self.data.targets[indx]
        else:
            self.samples = self.data.data
            self.labels = self.data.targets

        self.random_seed = 42

    def get_data(self):
        return self.samples, self.labels

    def partition(self, to_partition, indices, nr_agents):
        """
        Separate data into number of agents

        :param to_partition:        data to partition
        :param indices:             indices by which to partition
        :param nr_agents:           number of partitions to create
        :return:                    list of partitioned data
        """
        return [to_partition[indices[i]:indices[i + 1]] for i in range(nr_agents)]

    def split(self, how, nr_agents, **kwargs):
        """
        Different ways to split data between nodes.

        :param how:                 ['random', 'uniform', 'non_iid_uniform', 'non_iid_random']
        :param nr_agents:           number of agents to split data between
        :param kwargs:              additional class_per_node argument for non_iid case
        :return:                    data partitioned according to :param how.
        """
        if how == 'random':
            self.random_split(nr_agents)
        elif how == 'uniform':
            self.uniform_split(nr_agents)
        elif how == 'non_iid_uniform':
            self.non_iid_split(nr_agents, kwargs['class_per_node'], random=False)
        elif how == 'non_iid_random':
            self.non_iid_split(nr_agents, kwargs['class_per_node'], random=True)

        return self.get_data()

    def random_split(self, nr_agents):
        """
        Give each Node random splits of data. Nodes will have different amounts of data.

        :param nr_agents:
        """
        np.random.seed(self.random_seed)
        # Get random indices
        indices = sorted(np.random.randint(0, high=self.samples.shape[0], size=nr_agents - 1).tolist())
        indices = [0] + indices
        indices += [self.samples.shape[0]]

        self.samples = self.partition(self.samples, indices, nr_agents)
        self.labels = self.partition(self.labels, indices, nr_agents)

    def uniform_split(self, nr_agents):
        """
        Give each Node uniform splits of data. Nodes will have same amounts of data.

        :param nr_agents:
        """
        indices = np.linspace(start=0, stop=self.samples.shape[0], num=nr_agents + 1, dtype=int).tolist()

        self.samples = self.partition(self.samples, indices, nr_agents)
        self.labels = self.partition(self.labels, indices, nr_agents)

    def non_iid_split(self, nr_agents, class_per_node, random):
        """
        Give nodes only certain number of class labels as data.

        :param nr_agents:
        :param class_per_node:      number of class labels per node
        :param random:              boolean denoting random or uniform split of data
        """
        unique = list(set(self.labels.tolist()))
        len_unique = len(unique)

        # Create array that assigns a class to specific nodes
        # Use 'np.arange' to ensure every class is represented before repeating
        # A row represents nr_agents, a column represents classes per node
        agent_class_master = np.arange(start=0, stop=nr_agents * class_per_node) % len_unique
        np.random.shuffle(agent_class_master)
        agent_class_master = agent_class_master.reshape(nr_agents, class_per_node)

        # Split data by labels
        sample_list = [[] for _ in range(len_unique)]
        for i in range(len(self.labels)):
            sample_list[self.labels[i]].append(self.samples[i])

        # By class creates uniform or random indices splits to partition data to agents evenly
        class_count = np.bincount(agent_class_master.ravel())
        class_indices = {}
        for i in range(len(class_count)):
            if random:
                indices = sorted(np.random.randint(0, high=len(sample_list[i]), size=class_count[i] - 1).tolist())
                indices = [0] + indices
                indices += [len(sample_list[i])]
                class_indices[i] = indices
            else:
                class_indices[i] = np.linspace(start=0, stop=len(sample_list[i]), num=class_count[i] + 1,
                                               dtype=int).tolist()

        # Main loop that partitions data by the assigned class and proper amount
        all_agents = []
        all_class = []
        for agent in agent_class_master:
            agent_data = []
            agent_class = []
            for cls in agent:
                # Proportioned indices for data and grab correctly indexed data
                temp_indices = class_indices[cls]
                data_for_agent = sample_list[cls][temp_indices[0]:temp_indices[1] - 1]

                # Add data and class to this agents list
                agent_data = agent_data + data_for_agent
                agent_class = agent_class + [cls for _ in range(len(data_for_agent))]

                # Drop first index since we used that data, forces next person to use next index
                class_indices[cls] = temp_indices[1:]

            # Append agents data and class labels in order
            all_agents.append(torch.stack(agent_data))
            all_class.append(torch.tensor(agent_class))

        self.samples = all_agents
        self.labels = all_class


class CustomDataset(Dataset):
    """
    Creates dataset with both boolean and class labels.
    """

    def __init__(self, inputs, targets):
        """
        :param inputs:              Images 2x14x14
        :param targets:             Class Label
        """
        super(Dataset, self).__init__()
        self.data = []
        for inp, tgt in zip(inputs, targets):
            self.data.append([inp, tgt])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


def load_mnist_data(nr_nodes, nr_classes, allocation, subset, batch_size):
    """
    Uses LoadData class to partition prepare data. Puts data into dataloader objects to make use of batching and shuffling.

    :param nr_nodes:
    :param nr_classes:              if non_iid data, number class labels per node
    :param allocation:              ['random', 'uniform', 'non_iid_uniform', 'non_iid_random'] how to split data
    :param subset:                  subset of data
    :param batch_size:
    :return:
    """
    train_loader_list = []
    test_loader_list = []

    train = LoadData('MNIST', True, subset)
    test = LoadData('MNIST', False, False)

    train_data, train_targets = train.split(allocation, nr_nodes, class_per_node=nr_classes)
    for data, targets in zip(train_data, train_targets):
        train_dataset = CustomDataset(data, targets)
        train_loader_list.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))

    test_data, test_targets = test.split('uniform', nr_nodes)
    for data, targets in zip(test_data, test_targets):
        test_dataset = CustomDataset(data, targets)
        test_loader_list.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

    return train_loader_list, test_loader_list


def plot_mnist(data):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        a = data[i]
        pyplot.imshow(data[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
