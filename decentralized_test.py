import sys
from decentralized_network import DecentralizedNetwork

def main():
    # initialize
    dn = DecentralizedNetwork(
        20,
        3,
        'non_iid_uniform',
        "CycleGraph",
        0.9,
        0.0001,
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
