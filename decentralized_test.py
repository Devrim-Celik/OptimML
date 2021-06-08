import sys
from decentralized_network import DecentralizedNetwork

def main():
    # initialize
    dn = DecentralizedNetwork(
        3,
        3,
        'uniform',
        "CycleGraph",
        0.9,
        0.0001,
        10,
        "Adam",
        "MNIST",
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
