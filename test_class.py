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
        lr_list,
        alpha_list,
        training_epochs
    ):
        self.graph_list = graph_list
        self.task_list = task_list
        self.nr_node_list = nr_node_list
        self.nr_classes_list = nr_classes_list
        self.lr_list = lr_list
        self.alpha_list = alpha_list
        self.training_epochs = training_epochs

    def run(self):
        self.all_tests = []
        for test_indx in range(len(self.graph_list)):
            # create dictionary for saving values
            result_dic = {}
            # create decentralized network
            dn = DecentralizedNetwork(
                self.nr_node_list[test_indx],
                self.nr_classes_list[test_indx],
                'non_iid_uniform',
                self.graph_list[test_indx],
                self.alpha_list[test_indx],
                self.lr_list[test_indx],
                self.training_epochs,
                "Adam",
                self.task_list[test_indx]
            )
            # train it
            dn.train()
            # fill the dictionary
            result_dic["task"] = dn.task_type
            result_dic["graph"] = dn.graph_type
            result_dic["nr_classes"] = dn.nr_classes
            result_dic["nr_nodes"] = dn.nr_nodes
            result_dic["training_epochs"] = self.training_epochs

            # fill node specific information
            for node_indx in range(self.nr_node_list[test_indx]):
                node_dic = {}
                node_dic["lr"] = dn.nodes[node_indx].learning_rate
                node_dic["alpha"] = dn.nodes[node_indx].alpha
                node_dic["test_accuracies"] = dn.test_accuracies_nodes[node_indx]
                node_dic["test_losses"] = dn.test_losses_nodes[node_indx]
                node_dic["sent_bytes"] = dn.nodes[node_indx].sent_bytes
                node_dic["received_bytes"] = dn.nodes[node_indx].received_bytes
                result_dic[f"node_{node_indx}"] = node_dic

            # append to the total results
            self.all_tests.append(result_dic)

    def save_results(self, path):
        # save the results as a piickle, wir current time and date
        time_str = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        with open(f"{path}TestSuite-{time_str}.pkl", 'wb') as f:
            pickle.dump(self.all_tests, f, pickle.HIGHEST_PROTOCOL)
