# Custom Implemented Neural Network
---
This is a hobby project that I plan on expanding on to classify the MNIST handwritten digit dataset with. This is my own custom-implemented version of a neural-network, utilizing an architecture that consists of tensor batches of doubly linked link lists to reduce the overhead of pythonic processes. The goal is to make the most use out of numpy's vectorize operations as possible.

## Current Features
---
**Note that this list will be expanded**

  - **Data Handler Class**: contains useful functions such as train-test-split, standardization, etc. Taken directly from my logistic regression project.
  - **Layer**: Constructs Fully Connected (Dense) Layers, with base initializations. Supports ReLU, tanh, and sigmoid activations, with softmax or regression classification (mse or softmax)
  - **NeuralNetwork:** Strings together multiple tensors of specified batch size (unspecified is 32) and links them together with a doubly linked list. forward pass and backpropagation are here as well
    - **Features:** 
      - **Adam optimizer** (default)
      - **AdamW optimizer** (Adam with weight decay) (specify in train as string literal 'AdamW')
      - **train**: trains the model based on the input train and test data. here you can specify hyperparameters.
      - **test**: returns the loss and dataset of classified labels.
      - **adjustBatches**: manually adjust batch sizes for entire neural network.

**NOTE:** Features are subject to change, as this is a work in progress. Dropout and ease of use updates will be posted here.

## Dataset
---
The dataset can be found [here](https://www.kaggle.com/datasets/marshuu/breast-cancer).