from test_class import TestSuite

if __name__=="__main__":
    ts = TestSuite(
        graph_list = ["CycleGraph"],
        task_list = ["MNIST"],
        nr_node_list = [2],
        nr_classes_list = [3],
        lr_list = [0.0001],
        alpha_list = [0.9],
        training_epochs = 301
    )
    # run the tests
    ts.run()
    # save results
    ts.save_results("./results/")
