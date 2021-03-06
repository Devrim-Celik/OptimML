import pickle
from datetime import datetime

from decentralized_network import DecentralizedNetwork


class TestSuite:
    def __init__(
            self,
            graph_list,
            task_list,
            nr_node_list,
            nr_classes_list,
            data_distribution_list,
            lr_list,
            training_epochs,
            test_granularity,
            add_privacy_list,
            epsilon_list,
            delta_list,
            subset,
            batch_size
    ):
        self.graph_list = graph_list
        self.task_list = task_list
        self.nr_node_list = nr_node_list
        self.nr_classes_list = nr_classes_list
        self.data_distribution_list = data_distribution_list
        self.lr_list = lr_list
        self.training_epochs = training_epochs
        self.test_granularity = test_granularity
        self.add_privacy_list = add_privacy_list
        self.epsilon_list = epsilon_list
        self.delta_list = delta_list
        self.subset = subset
        self.batch_size = batch_size

    def run(self):
        self.all_tests = []
        for test_indx in range(len(self.graph_list)):
            self.print_header(test_indx)
            result_dic = {}
            dn = DecentralizedNetwork(
                self.nr_node_list[test_indx],
                self.nr_classes_list[test_indx],
                self.data_distribution_list[test_indx],
                self.graph_list[test_indx],
                self.lr_list[test_indx],
                self.training_epochs[test_indx],
                "Adam",
                self.task_list[test_indx],
                self.add_privacy_list[test_indx],
                self.epsilon_list[test_indx],
                self.delta_list[test_indx],
                self.subset[test_indx],
                self.batch_size[test_indx],
                test_granularity=self.test_granularity[test_indx]
            )

            # Train the decentralized network
            dn.train()

            # Fill the dictionary
            result_dic["task"] = dn.task_type
            result_dic["graph"] = dn.graph_type
            result_dic["nr_classes"] = dn.nr_classes
            result_dic["nr_nodes"] = dn.nr_nodes
            result_dic["training_epochs"] = self.training_epochs
            result_dic["epoch_list"] = dn.epoch_list
            result_dic["add_privacy_list"] = dn.add_privacy
            result_dic["epsilon_list"] = dn.epsilon
            result_dic["delta_list"] = dn.delta
            result_dic["batch_size"] = dn.batch_size

            # Fill node specific information
            for node_indx in range(self.nr_node_list[test_indx]):
                node_dic = {}
                node_dic["lr"] = dn.nodes[node_indx].learning_rate
                node_dic["test_accuracies"] = [row[node_indx] for row in dn.test_accuracies_nodes]
                node_dic["test_losses"] = [row[node_indx] for row in dn.test_losses_nodes]
                node_dic["train_accuracies"] = dn.nodes[node_indx].train_accuracies
                node_dic["train_losses"] = dn.nodes[node_indx].train_losses
                node_dic["sent_bytes"] = [row[node_indx] for row in dn.sent_bits]
                node_dic["received_bytes"] = [row[node_indx] for row in dn.received_bits]
                result_dic[f"node_{node_indx}"] = node_dic

            # Append to the total results
            self.all_tests.append(result_dic)
            self.save_results("./results/")

    def save_results(self, path):
        # Save the results as a pickle, with current time and date
        time_str = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        with open(f"{path}TestSuite-{time_str}.pkl", 'wb') as f:
            pickle.dump(self.all_tests, f, pickle.HIGHEST_PROTOCOL)

    def print_header(self, index):
        print(f"====== Scenario {index} ======")
        print(f"GRAPH: {self.graph_list[index]} | NUM NODES: {self.nr_node_list[index]} | DATA DIST: {self.data_distribution_list[index]}")
        print(f"DATA: {self.task_list[index]} | LR: {self.lr_list[index]} | PRIVACY: {self.add_privacy_list[index]} | BATCH: {self.batch_size[index]}")
        print("\n")