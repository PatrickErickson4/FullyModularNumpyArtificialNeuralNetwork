'''
@author: Patrick Erickson
Project Name: Layer
Project Description: Constructs a modular implementation a fully connected layer
'''
import numpy as np

__all__ = [
    '__init__',
]

class FullyConnectedLayer:
    '''
    This class creates a modular Fully Connected Layer that allows you to customize the number of nodes, activation, and dropout. Uses
    lambdas to assign functions to obtain the derivatives with respect to the activation functions

    FUNCTIONS:
    __init__: constructs the Layer
    adjustBatchSize: readjusts the batch size for the specific layer, rebroadcasting necessary data values
    newMoment: using the formula from the adam paper, constructs and computes the moment for the specific layer
    biasCorrected: takes in moments and applies the bias corrections as mentioned in the adam paper
    _softmax: helper computes a stable softmax for categorical cross entropy lambda
    _stableSigmoid: helper that computes a stable sigmoid to ensure convergence for this activation
    _sigmoidDerivative: helper that computes the derivative of a stable sigmoid to ensure convergence for this activation
    _defineActivation: helper for initializer that assigns the correct list of identifiers and lambda functions based on the activation parameter in initializer
    '''

    def __init__(self,numNodes,activation, dropout = 0):
        '''
        Constructs a FullyConnectedLayer object. Placeholders are put in for the batch size.
        Performs basic checks on correct param arguments.

        INPUTS:
        numNodes: The number of neurons specified by the user for this input layer
        activation: the activation function wanting to be used for the layer
        dropout: the probability under binomial for the dropout mask
        '''

        if numNodes <= 0:
            raise Exception("Error in initialization for FullyConnectedLayer object: Enter a valid feature size.")
        if dropout >= 1 or dropout < 0:
            raise Exception("Pick better dropout foo (0 < dropout < 1).")
        if dropout != 0 and (activation.lower() == 'softmax' or activation.lower() == 'mse'):
            raise Exception("Do not perform dropout on final layer.")

        self.dropout = dropout
        self.mask = [1]
        self.featureSize = numNodes
        self.inputs = np.zeros((1,numNodes,1))
        self.activated = np.zeros((1,numNodes,1))
            
        self.activation = self._defineActivation(activation)
        self.weights = None
        self.bias = None

        self.m = None # moment
        self.v = None # second moment
        self.mBias = None # moment
        self.vBias = None # second moment

        self.next = None
        self.prev = None
    
    def adjustBatchSize(self, newBatchSize):
        '''
        Changes the batch size for the layer. Can
        once the object is made. Calls on 
        node's adjust batch size to change specific node batches for weights

        INPUT:
        newBatchSize: client-specified batch. Never called directly.
        '''

        # rebroadcasts with correct batch size
        self.weights = np.repeat(self.weights[:,:,0:1], newBatchSize, axis=2)
        self.bias = np.repeat(self.bias[:,:,0:1], newBatchSize, axis=2)
        self.inputs = np.repeat(self.inputs[:,:,0:1], newBatchSize, axis=2)
        self.activated = np.repeat(self.activated[:,:,0:1], newBatchSize, axis=2)

    def newMoment(self, grad, biasGrad, Beta1=.9,Beta2=.999):

        '''
        Computes recursively the new moment and its moving average based on the gradient for that layer.
        NOTE: this is layer dependent, and each layer will have its own moment.

        INPUT:
        grad: calculated gradient of the weights for the layer
        biasGrad: calculated loss w/r to the gradients
        Beta1: constant, conventionally .9
        Beta2: constant, conventionally .999
        '''
        self.m = Beta1*self.m + (1-Beta1)*grad
        self.v = Beta2*self.v + (1-Beta2)*(grad**2)
        self.mBias = Beta1*self.mBias + (1-Beta1)*biasGrad
        self.vBias = Beta2*self.vBias + (1-Beta2)*(biasGrad**2)
        return self.m, self.v, self.mBias, self.vBias
    
    def biasCorrected(self,m1,m2,m1Bias,m2Bias,iterations,Beta1=.9,Beta2=.999,):
        '''
        Bias corrects the new moments and its moving averages so that when first calculating gradients, we aren't biased towards 0.
        NOTE: this is layer dependent, and each layer will have its own moment.

        INPUT:
        m1: moment
        m2: moving average of moment
        m1Bias: moment for bias
        m2Bias: moving average of moment for bias
        Beta1: constant, conventionally .9
        Beta2: constant, conventionally .999
        '''
        return m1/(1-Beta1**iterations), m2/(1-Beta2**iterations),m1Bias/(1-Beta1**iterations), m2Bias/(1-Beta2**iterations)


    def _softmax(self,x):
        '''
        Helper function for computing softmax. Used by activation for lambda

        INPUT:
        x: lambda's input
        '''
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)
    
    def _stableSigmoid(self,x): # taken from logistic regression
        '''
        Helper function for computing stable sigmoid. Used by activation for lambda

        INPUT:
        x: lambda's input
        '''
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))
    def _sigmoidDerivative(self,x):
        '''
        Helper function for computing stable sigmoid's derivative. Used by activation for lambda

        INPUT:
        x: lambda's input
        '''
        s = self._stableSigmoid(x)
        return s * (1 - s)


    def _defineActivation(self, activation):
        '''
        Assigns the correct list of activations, their derivatives, their identity, and if permitted their log loss.

        INPUT:
        activation: String of the activation wanting to be used.

        OUTPUT:
        Array of: 
            lambda functions: contains the activation, its derivative, and if its an end layer (softmax, mse), its log loss.
            identifier: the string identifier, changed to lowercase. 
            NOTE: These are not in order.
        '''

        #activations and derivatives of inner layers
        if activation.lower() == 'linear':
            return lambda x: x, lambda x: np.ones_like(x), 'linear'
        elif activation.lower() == 'relu':
            return lambda x: np.where(x > 0, x, 0), lambda x: np.where(x > 0, 1, 0), 'relu'
        elif activation.lower() == 'tanh':
            return lambda x: np.tanh(x), lambda x: 1-np.tanh(x)**2, 'tanh'
        elif activation.lower() == 'sigmoid':
            return lambda x: self._stableSigmoid(x) , lambda x: self._sigmoidDerivative(x), 'sigmoid'
        
        #activations and derivatives of final heads for loss and gradient w/r to loss
        elif activation.lower() == 'mse':
            return lambda x: x, lambda x,y: x-y, 'mse', lambda x,y: np.mean(np.sum(np.square(x - y), axis=1))
        elif activation.lower() == 'softmax':
            return lambda x: self._softmax(x), lambda x, y: (x - y).reshape(x.shape),'softmax', lambda x,y: -np.mean(np.sum(np.multiply(y, np.log(x + 1e-8)), axis=1))
        else:
            raise Exception("Enter a valid activation function.")
    
