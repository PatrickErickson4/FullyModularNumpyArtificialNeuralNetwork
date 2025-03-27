# Custom Implemented Neural Network
---
This is a hobby project that I plan on expanding on to classify the MNIST handwritten digit dataset with. This is my own custom-implemented version of a neural-network, utilizing an architecture that consists of tensor batches of doubly linked link lists to reduce the overhead of pythonic processes. The goal is to make the most use out of numpy's vectorize operations as possible. Features were constructed with the use of a basic logistic regression dataset, then validated with MNIST. 

## Current Features
---
**Note that this list will be expanded**

  - **Data Handler Class**: contains useful functions such as train-test-split, standardization, etc. Taken directly from my logistic regression project.
  - **Layer**: Constructs Fully Connected (Dense) Layers, with base initializations. Supports ReLU, tanh, and sigmoid activations, with softmax or regression classification (mse or softmax)
  - **NeuralNetwork:** Strings together multiple tensors of specified batch size (unspecified is 32) and links them together with a doubly linked list. forward pass and backpropagation are here as well
    - **Modular Features:** 
      - **Individual Modularized Layers**: Construct Standalone layers with ```FullyConnectedLayer(numNodes,activation,dropout=p)```. Note that the input layer is constructed for you.
      - **Specifiable Layer architecture**: after specifying batch size and input dropout (optional), ```**kwargs``` allows for any number of ```layer1 = FullyConnectedLayer(),layer2 = FullyConnectedLayer()```.   
        - **NOTE:** The last layer must be either mse or softmax.
      - **Adam optimizer**: (default)
      - **AdamW optimizer**: (Adam with weight decay) (specify in train as string literal 'AdamW' in ```NeuralNetwork.train(loss='AdamW',weightDecay=%)```)
      - **train**: trains the model based on the input train and test data. here you can specify hyperparameters.
      - **test**: returns the loss and dataset of classified labels.
      - **adjustBatches**: manually adjust batch sizes for entire neural network. (```NeuralNetwork.adjustBatchSize(newBatchSize)```)
      - **Input Dropout**: Can be specified in model initialization. Drops specified percentage of input features via a binomial mask for more robust training. (Specify as a parameter in ```NeuralNetwork(inputDropout = p)```)
      - **Dropout**: Implemented a per-layer dropout. (Specify as a parameter in individual ```FullyConnectedLayer(dropout=p)```)


**NOTE:** Features are subject to change. More explicit docstrings will be added later. A saved model will be available for download as a JSON file.

**NOTE:** Numpy does not utilize GPU. In addition to the pythonic nature of this implementation, training will be much slower than conventional state-of-the-art libraries. You can probably graduate college, get married, have a child, and retire by the time you're finished training anything over 100,000 cases.
## Dataset
---
The dataset used for debugging and batch dimension checking be found [here](https://www.kaggle.com/datasets/marshuu/breast-cancer).

The final implementation uses MNIST, found [here](http://yann.lecun.com/exdb/mnist/).