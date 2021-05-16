import sys
from decentralized_network import DecentralizedNetwork

def main():
    # initialize
    dn = DecentralizedNetwork(
        1,
        10,
        'non_iid_uniform',
        "FullyConnectedGraph",
        0.9,
        0.001,
        1000,
        "Adam",
        "MNIST"
    )

    dn.train()
    dn.plot_training()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
