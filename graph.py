import abc
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Topology(abc.ABC):
    """
    Base class to create various network topologies.
    """
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.graph = self._create_graph()

    @property
    def get_n_nodes(self):
        return self.n_nodes

    @property
    @abc.abstractmethod
    def get_n_edges(self):
        pass

    @abc.abstractmethod
    def _create_graph(self):
        pass

    def adj(self):
        """
        Get concise form of nodes and their neighbors.
        :return: dict {node : list of connection to node}
        """
        graph_no_diag = self.graph - np.identity(self.n_nodes)
        adj_list = {}
        for node in range(self.n_nodes):
            adj = np.nonzero(graph_no_diag[node])[0]
            adj_list[node] = adj
        return adj_list

    def __getitem__(self, node):
        """
        :param node: node id
        :return: dict {neighbor node : connection strength}
        """
        graph_no_diag = self.graph - np.identity(self.n_nodes)
        adj = graph_no_diag[node]
        adj_index = np.nonzero(adj)[0]
        return {k: 1 for k in adj_index}


class FullyConnectedGraph(Topology):
    def __init__(self, n_nodes):
        super().__init__(n_nodes)

    def _create_graph(self):
        return np.ones((self.n_nodes, self.n_nodes))

    def get_n_edges(self):
        return self.n_nodes * (self.n_nodes - 1) / 2


def build_adj_dict(graph):
    """
    Helper to build dict with node and neighboring nodes as list.
    :param graph: nx.Graph objects
    :return: dict(node : [list of neighbors]}
    """
    adj_dict = {}
    for node in graph.nodes():
        nb_list = []
        for nb in graph.neighbors(node):
            nb_list.append(nb)
        adj_dict[node] = nb_list
    return adj_dict


# Helper function for printing various graph properties
def describe_graph(G):
    print(nx.info(G))
    print("Avg. Shortest Path Length: %.4f" %nx.average_shortest_path_length(G))
    print("Diameter: %.4f" %nx.diameter(G)) # Longest shortest path
    print("Sparsity: %.4f" %nx.density(G))  # #edges/#edges-complete-graph
    print("Degree Histogram: {}".format({k: v for (k, v) in zip(range(len(nx.degree_histogram(G))), nx.degree_histogram(G))}))


def plot_degree_distribution(G):
    degrees = {}
    for node in G.nodes():
        degree = G.degree(node)
        if degree not in degrees:
            degrees[degree] = 0
        degrees[degree] += 1
    sorted_degree = sorted(degrees.items())
    deg = [k for (k,v) in sorted_degree]
    cnt = [v for (k,v) in sorted_degree]
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title("Degree Distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Degree")
    ax.set_xticks([d+0.05 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()


def visualize_graph(G, with_labels=True, k=None, alpha=1.0, node_shape='o'):
    pos = nx.spring_layout(G, k=k)
    if with_labels:
        lab = nx.draw_networkx_labels(G, pos, labels=dict([(n, n) for n in G.nodes()]))
    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='g', node_shape=node_shape)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    graph = nx.cycle_graph(10)
    graph2 = nx.ring_of_cliques(10, 4)
    graph3 = nx.binomial_graph(10, .2, seed=1)
    graph4 = nx.circulant_graph(20, range(1, 10))
    adj1 = build_adj_dict(graph)
    adj2 = build_adj_dict(graph2)
    adj3 = build_adj_dict(graph3)
    adj4 = build_adj_dict(graph4)
    visualize_graph(graph4)