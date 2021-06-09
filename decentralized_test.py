import sys
from decentralized_network import DecentralizedNetwork

def main():
    # initialize
    dn = DecentralizedNetwork(
        nr_nodes=3,
        nr_classes = 3,
        allocation = 'uniform',
        graph_type = "CycleGraph",
        alpha = 0.9,
        lr = 0.00001,
        training_epochs = 10,
        optimizer_type = "Adam",
        task_type = "MNIST",
        add_privacy = False,
        epsilon = 1,
        delta = 1,
        test_granularity = 1
    )
    dn.train()
    dn.plot_training()
    print(dn.get_bytes())
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
