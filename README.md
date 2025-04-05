# Fully Modular Numpy-restricted Neural Network with Adam Optimization and Regularization
---

This is a hobby project of mine that I used to classify the MNIST handwritten digit dataset with. This is my own custom-implemented version of a neural-network, utilizing a custom architecture that consists of tensor batches of doubly linked link lists in an attempt to reduce the overhead of pythonic processes. The architecture consists of chained tensors of weight matrices multiplied by the dimensions of their batches, which are then fed forward and back propagated, with einsums to simplify logic of transposes and matmuls. Adaptive moments and the moving averages of these moments are used to correct the descent path for a smoother convergence. The goal with this architecture to make the most use out of numpy's vectorize operations as possible, while still maintaining some semblance of interpretability. Features were constructed with the use of a basic logistic regression dataset, then validated with MNIST. This validation was able to produce up to a **97% test accuracy**. MNIST h5 models trained on this neural network architecture are provided. You can reproduce it running the modelConstructor.ipynb file (**WARNING: VERY SLOW**), and mess around with it in drawer.py.


**NOTE**: You may be confused as to why you see tensorflow. Tensorflow is used solely to download the MNIST handwritten digits dataset.

## Features
---
  - **NeuralNetworkScript**:
    - **Data Handler Class**: contains useful functions such as train-test-split, standardization, a one-hot encoder, and unnormalization. Taken directly from my logistic regression project, aside from the one-hot encoding.
    - **Layer**: Constructs Fully Connected (Dense) Layers, with base initializations. Supports ReLU, tanh, and sigmoid activations, with regression or classification (mse or softmax) as a final layer.
    - **Neural Network:** Strings together multiple tensors of specified batch size (unspecified is 32) and links them together with a doubly linked list. forward pass and backpropagation are here as well.
      - **Modular Features:** 
        - **Individual Modularized Layers**: Construct Standalone layers with ```FullyConnectedLayer(numNodes,activation,dropout=percentProbability)```. Note that the input layer is constructed for you.
        - **Specifiable Layer architecture**: after specifying batch size and input dropout (optional), ```**kwargs``` allows for any number of ```layer1 = FullyConnectedLayer(),layer2 = FullyConnectedLayer()```.   
          - **NOTE:** The last layer must be either mse or softmax.
        - **Adam optimizer**: (default)
        - **AdamW optimizer**: (Adam with weight decay) (specify in train as string literal 'AdamW' in ```NeuralNetwork.train(loss='AdamW',weightDecay=nonZeroValue)```)
        - **train**: trains the model based on the input train and test data. here you can specify hyperparameters.
        - **test**: returns the loss and dataset of classified labels.
        - **adjustBatches**: manually adjust batch sizes for entire neural network. (```NeuralNetwork.adjustBatchSize(newBatchSize)```)
        - **Input Dropout**: Can be specified in model initialization. Drops specified percentage of input features via a binomial mask for more robust training. (Specify as a parameter in ```NeuralNetwork(inputDropout = percentProbability)```)
        - **Dropout**: Implemented a per-layer dropout. (Specify as a parameter in individual ```FullyConnectedLayer(dropout=percentProbability)```)
        - **saveModel:** Implemented a model saving feature. Models can be saved doing ```model.saveModel('fileName')```. Accessing the model can be done by doing ```savedModel = NeuralNetwork(model='fileName')```. Do not include extensions for either file name.
  - **Interactives**:
    - **tutorial.ipynb**: shows you how to implement the Neural Network
    - **modelConstructor.ipynb**: the file in which I trained the MNIST models in. Feel free to try and find one yourself!
    - **drawing.py**: An AI coding assisted .py file that utilizes the models trained, where you can draw digits and have the models try to predict them! (It kind of blows, it likes the number 6 a lot. This is to be expected though, since MNIST is super clean data with not a lot of variation). Can be found in the outermost layer of the directory.



**NOTE:** Numpy does not utilize GPU. In addition to the pythonic nature of this implementation, training will be much slower than conventional state-of-the-art libraries. You can probably graduate college, get married, have a child, and retire by the time you're finished training anything over 100,000 cases.
## Dataset
---

The dataset used for debugging and batch dimension checking be found [here](https://www.kaggle.com/datasets/marshuu/breast-cancer).

The final implementation uses MNIST, found [here](http://yann.lecun.com/exdb/mnist/).

## Setup
---

Ensure that you have some sort of python environment installed. Also make sure you have git installed. You can go [here](https://git-scm.com/downloads/win) to install git.

  - **Step 1**: Create a folder you want the file to be.
  - **Step 2**: Open your command line of choice (powershell, cmd, mac alternatives, linux, etc.)
  - **Step 2**: Navigate to your folder with ```cd yourFolderName```
  - **Step 3**: Clone the repository with ```git clone https://github.com/PatrickErickson4/FullyModularNumpyArtificialNeuralNetwork.git```
  - **Step 4**: Install the dependencies with ```pip install -r requirements.txt```

## Starting out
---
  - **Step 1**: Open InteractivesAndModels
  - **Step 2**: Click on the tutorial and look through it if you want to implement it yourself!
  - **Step 3**: Check out how well the model generalizes to new data by running driver.py! (It does poorly lol)

## License
---

The following was produced under the MIT license.
