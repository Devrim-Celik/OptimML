Decentralized Machine Learning
======================================================================

This repository provides a framework for building network topologies and testing decentralized machine learning models.

Accompanying report: REPORT LINK

## Software dependencies

The code contained in this repository was tested on the following configuration of Python:

- numpy==1.19.5
- torchvision==0.9.1
- matplotlib==3.4.1
- opacus==0.13.0
- sklearn==0.0
- scikit-learn==0.24.2
- networkx==2.5.1
- prettytable==2.1.0

## Installation Guide

```bash
pip3 install -r requirements.txt
```

## Running our Code

Run the run.py file in the notebook to start running scenarios. To change which tests are run edit the parameter dictionary.
See next section for an explanation of parameters. The default parameters we have left correspond to ...

 - [run.py](run.py)

## Run Parameters Explained
For running a test you specify all the parameters in lists. We then take all the permutations of these parameters and run these sequentially.

```[Potential options listed in a list]```

**graph_list:** ```["FullyConnectedGraph", "BinomialGraph", "RingOfCliques", "CirculantGraph", "CycleGraph", "Torus2D"]``` Choice of network topology. This decides how the nodes are laid out and who communicates with who. <br/>
**task_list:** ```["MNIST"]``` Which dataset to use. Currently only implemented MNIST <br/>
**nr_node_list:**  ```[Natural Numbers]``` Number of nodes to create. Note certain network topologies can only have a certain number of nodes. <br/>
**nr_classes_list:** ```[0-10] for MNIST``` Only has an effect if data_distribution is "non_iid". Determines the number of class labels that are given to each node. <br/>
**data_distribution:** ```['uniform', 'random', 'non_iid_uniform', 'non_iid_random']``` How data is split across nodes. Uniform means that nodes are given the same number of samples, random means the samples are randomly partitioned. non_iid means we do not shuffle the data and give each node only a certain number of class labels as defined by **nr_classes_list**.<br/>
**lr_list:** ```[Real Numbers]``` learning rate<br/>
**training_epochs:** ```[Natural Numbers]``` number of training epochs to run<br/>
**test_granularity:** ```[Natural Numbers]``` frequency with which to test the network on the test data set. Corresponds to test_granularity % epochs<br/>
**add_privacy_list:** ```[Boolean]``` Boolean flag to add differential privacy<br/>
**epsilon_list:** ```[Real Numbers]``` Only has effect if add_privacy_list is True. Quantifies the privacy properties of the DP-SGD algorithm. More [info](https://opacus.ai/docs/faq). <br/>
**delta_list:** ```[Real Numbers]``` Only has effect if add_privacy_list is True. Quantifies the privacy properties of the DP-SGD algorithm. More [info](https://opacus.ai/docs/faq). <br/>
**subset:** ```[Boolean]``` Whether or not to train on 30% of the training data. Used to save time. <br/>

## Plots and Results from Paper

- [plots.ipynb](plots.ipynb)

## File Structure
Here is the file structure of the project:
```bash
Project
|
|-- data -- |
|   |-- MNIST -- |
|       |-- processed
|       |-- raw
|-- plots -- |
|-- results -- |
|-- .gitignore
|-- auxiliary.py
|-- data.py
|-- decentralized_network.py
|-- decentralized_test.py
|-- graph.py
|-- network_test.py
|-- network.py
|-- node.py
|-- plots.ipynb
|-- README.md
|-- run.py
|-- test_class.py

```

## Files
* `auxiliary.py`: functions for counting parameters of a model and loading data
* `data.py`: class for generating data and bringing it in adequate form for the setup at hand
* `decentralized_network.py`: class representing the complete decentralized network, encompassing many nodes from `node.py`
* `decentralized_test.py`: test suite for decentralized network class
* `graph.py`: graph topology classes
* `network_test.py`: test suite for testing network class from `network.py`
* `network.py`: network class, used in the node classes
* `node.py`: classes for nodes; each one represents one agent in the decentralized network
* `plots.ipynb`: for creating the plots
* `run.py`: for executing multiple runs
* `test_class.py`: class for creating multiple test setups and saving the results


- Authors: Alec Flowers, Devrim Celik, Nina Mainusch
