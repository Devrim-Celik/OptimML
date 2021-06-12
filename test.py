from test_class import TestSuite
from sklearn.model_selection import ParameterGrid


def run_test():
    parameters = {"graph_list":["FullyConnectedGraph", "Torus2D", "CycleGraph", "RingOfCliques"],
         "task_list":["MNIST"],
         "nr_node_list":[4, 16, 32],
         "nr_classes_list":[0],
         "lr_list":[0.0001],
         "alpha_list":[0.5],
         "training_epochs":[10],
         "test_granularity":[1],
         "add_privacy_list":[False],
         "epsilon_list":[0.1],
         "delta_list":[.5],
         "subset":[True]}
    grid = []
    for p in ParameterGrid(parameters):
        grid.append(p)

    ts = TestSuite(
        graph_list=[val['graph_list'] for val in grid],
        task_list=[val['task_list'] for val in grid],
        nr_node_list=[val['nr_node_list'] for val in grid],
        nr_classes_list=[val['nr_classes_list'] for val in grid],
        lr_list=[val['lr_list'] for val in grid],
        alpha_list=[val['alpha_list'] for val in grid],
        training_epochs=[val['training_epochs'] for val in grid],
        test_granularity=[val['test_granularity'] for val in grid],
        add_privacy_list = [val['add_privacy_list'] for val in grid],
        epsilon_list = [val['epsilon_list'] for val in grid],
        delta_list = [val['delta_list'] for val in grid],
        subset = [val['subset'] for val in grid]
    )
    # run the tests
    ts.run()
    # save results
    ts.save_results("./results/")

if __name__=="__main__":
    run_test()