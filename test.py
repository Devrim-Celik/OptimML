from test_class import TestSuite

if __name__=="__main__":
    ts = TestSuite(
        graph_list=["FullyConnectedGraph", "FullyConnectedGraph", "FullyConnectedGraph"],
        task_list=["MNIST", "MNIST", "MNIST"],
        nr_node_list=[8, 16, 32],
        nr_classes_list=[0, 0, 0],
        lr_list=[0.001, 0.001, 0.001],
        alpha_list=[0.5, 0.5, 0.5],
        training_epochs=5,
        test_granularity=1,
        add_privacy_list = [False, False, False],
        epsilon_list = [0.1,0.1,0.1],
        delta_list = [0.5,0.5,0.5]
    )
    # run the tests
    ts.run()
    # save results
    ts.save_results("./results/")
