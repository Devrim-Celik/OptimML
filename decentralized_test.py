import sys
from decentralized_network import DecentralizedNetwork

def main():
    # initialize
    dn = DecentralizedNetwork(
        nr_nodes=4,
        nr_classes=3,
        allocation='random',
        graph_type="Torus2D",
        alpha=0.9,
        lr=0.1,
        training_epochs=20,
        optimizer_type="Adam",
        task_type="MNIST",
        add_privacy=True,
        epsilon=1.1,
        delta=1e-6,
        subset=False,
        batch_size=128,
        test_granularity=1
    )
    dn.train()
    dn.plot_training()
    print(dn.get_bytes())
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
