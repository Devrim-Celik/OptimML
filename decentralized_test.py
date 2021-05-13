import sys
from decentralized_network import DecentralizedNetwork

def main():
    # initialize
    dn = DecentralizedNetwork(
        2,
        "CycleGraph",
        0.9,
        0.01,
        1000,
        "Adam",
        "MNIST",
        True,
    )

    dn.train()
    dn.plot_training()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
