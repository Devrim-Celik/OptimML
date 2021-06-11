import abc
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

class Topology(abc.ABC):
    """
    Base class to create various network topologies.
    """
    def __init__(self):
        self.graph = self._create_graph()

    @property
    def n_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def n_edges(self):
        return self.graph.number_of_edges()

    @property
    def density(self):
        return nx.density(self.graph)

    @property
    def avg_degree_connectivity(self):
        return nx.average_degree_connectivity(self.graph)

    @abc.abstractmethod
    def _create_graph(self):
        pass

    def visualize_graph(self, with_labels=True, k=None, alpha=1.0, node_shape='o'):
        pos = nx.spring_layout(self.graph, k=k)
        if with_labels:
            lab = nx.draw_networkx_labels(self.graph, pos, labels=dict([(n, n) for n in self.graph.nodes()]))
        ec = nx.draw_networkx_edges(self.graph, pos, alpha=alpha)
        nc = nx.draw_networkx_nodes(self.graph, pos, nodelist=self.graph.nodes(), node_color='g', node_shape=node_shape)
        plt.axis('off')
        plt.show()

    def plot_degree_distribution(self):
        degrees = {}
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree not in degrees:
                degrees[degree] = 0
            degrees[degree] += 1
        sorted_degree = sorted(degrees.items())
        deg = [k for (k, v) in sorted_degree]
        cnt = [v for (k, v) in sorted_degree]
        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')
        plt.title("Degree Distribution")
        plt.ylabel("Frequency")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.05 for d in deg])
        ax.set_xticklabels(deg)
        plt.show()

    def adj(self):
        """
        Get concise form of nodes and their neighbors.
        :return: dict {node : list of connection to node}
        """
        adj_dict = {}
        for node in self.graph.nodes():
            nb_list = []
            for nb in self.graph.neighbors(node):
                nb_list.append(nb)
            adj_dict[node] = nb_list
        return adj_dict

    # def __getitem__(self, node):
    #     """
    #     :param node: node id
    #     :return: dict {neighbor node : connection strength}
    #     """
    #     graph_no_diag = self.graph - np.identity(self.n_nodes)
    #     adj = graph_no_diag[node]
    #     adj_index = np.nonzero(adj)[0]
    #     return {k: 1 for k in adj_index}


class FullyConnectedGraph(Topology):
    def __init__(self, nodes):
        self.nodes = nodes
        super().__init__()

    def _create_graph(self):
        return nx.complete_graph(self.nodes)


class BinomialGraph(Topology):
    def __init__(self, nodes, probability):
        """
        Initialise the graph with nodes (int), probability (float [0,1])
        """
        self.nodes = nodes
        self.probability = probability
        super().__init__()

    def _create_graph(self):
        return nx.binomial_graph(self.nodes, self.probability)


class RingOfCliques(Topology):
    """
    A ring of cliques graph is consisting of cliques, connected through single links. Each clique is a complete graph.
    """
    def __init__(self, num_cliques, clique_size=2):
        if num_cliques == 4:
            self.num_cliques = 2
            self.clique_size = 2
        elif num_cliques == 16:
            self.num_cliques = 4
            self.clique_size = 4
        elif num_cliques == 32:
            self.num_cliques = 4
            self.clique_size = 8
        else:
            self.num_cliques = num_cliques
            self.clique_size = clique_size

        super().__init__()

    def _create_graph(self):
        return nx.generators.community.ring_of_cliques(self.num_cliques, self.clique_size)


class CirculantGraph(Topology):
    """
    Generates the circulant graph Ci_n(x_1,x_2,...,x_m) with n vertices.
    """
    def __init__(self, n, offset: List[int]):
        """
        Initialise the graph with n (int), offset (list)
        """
        self.n = n
        self.offset = offset
        super().__init__()

    def _create_graph(self):
        return nx.generators.classic.circulant_graph(self.n, self.offset)


class CycleGraph(Topology):
    def __init__(self, n):
        """
        Initialise the graph with n (int)
        """
        self.n = n
        super().__init__()

    def _create_graph(self):
        return nx.cycle_graph(self.n)

class Torus2D(Topology):
    def __init__(self, m, n=2):
        if m == 4:
            self.m = 2
        elif m == 16:
            self.m = 8
        elif m == 32:
            self.m = 16
        else:
            self.m = m
        self.n = n
        super().__init__()

    def _create_graph(self):
        return nx.grid_2d_graph(self.m, self.n, periodic=True)

    def tuple_to_1Dindex(self, tuple_):
        return tuple_[0] * self.n + tuple_[1]

    def adj(self):
        """
        Get concise form of nodes and their neighbors.
        :return: dict {node : list of connection to node}
        """
        adj_dict = {}
        for node in self.graph.nodes():
            nb_list = []
            for nb in self.graph.neighbors(node):
                nb_list.append(self.tuple_to_1Dindex(nb))

            adj_dict[self.tuple_to_1Dindex(node)] = nb_list
        return adj_dict

if __name__ == '__main__':
    graph = CycleGraph(10)
    a = graph.n_nodes
    b = graph.adj()
    c = graph.visualize_graph()
    f = graph.plot_degree_distribution()
    c

    # graph = nx.cycle_graph(10)
    # graph2 = nx.ring_of_cliques(10, 4)
    # graph3 = nx.binomial_graph(10, .2, seed=1)
    # graph4 = nx.circulant_graph(20, range(1, 10))
    # adj1 = build_adj_dict(graph)
    # adj2 = build_adj_dict(graph2)
    # adj3 = build_adj_dict(graph3)
    # adj4 = build_adj_dict(graph4)
    # visualize_graph(graph4)
