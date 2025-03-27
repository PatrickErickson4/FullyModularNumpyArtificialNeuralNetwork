import numpy as np

class FullyConnectedLayer:

    def __init__(self,numNodes,activation, dropout = 0):

        if numNodes <= 0:
            raise Exception("Error in initializatoin for FullyConnectedLayer object: Enter a valid feature size.")
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
        self.weights = np.repeat(self.weights[:,:,0:1], newBatchSize, axis=2)
        self.bias = np.repeat(self.bias[:,:,0:1], newBatchSize, axis=2)
        self.inputs = np.repeat(self.inputs[:,:,0:1], newBatchSize, axis=2)
        self.activated = np.repeat(self.activated[:,:,0:1], newBatchSize, axis=2)


    def _softmax(self,x):
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)
    
    def _stableSigmoid(self,x): # taken from logistic regression
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))
    def _sigmoidGrad(self,x):
        s = self._stableSigmoid(x)
        return s * (1 - s)


    def _defineActivation(self, activation):

        #activations and derivatives of inner layers
        if activation.lower() == 'linear':
            return lambda x: x, lambda x: np.ones_like(x), 'linear'
        elif activation.lower() == 'relu':
            return lambda x: np.where(x > 0, x, 0), lambda x: np.where(x > 0, 1, 0), 'relu'
        elif activation.lower() == 'tanh':
            return lambda x: np.tanh(x), lambda x: 1-np.tanh(x)**2, 'tanh'
        elif activation.lower() == 'sigmoid':
            return lambda x: self._stableSigmoid(x) , lambda x: self._sigmoidGrad(x), 'sigmoid'
        
        #activations and derivatives of final heads for loss and gradient w/r to loss
        elif activation.lower() == 'mse':
            return lambda x: x, lambda x,y: x-y, 'mse', lambda x,y: np.mean(np.sum(np.square(x - y), axis=1))
        elif activation.lower() == 'softmax':
            return lambda x: self._softmax(x), lambda x, y: (x - y).reshape(x.shape),'softmax', lambda x,y: -np.mean(np.sum(np.multiply(y, np.log(x + 1e-8)), axis=1))
        else:
            raise Exception("Enter a valid activation function.")
        
    def newMoment(self, grad, biasGrad, Beta1=.9,Beta2=.999):
        self.m = Beta1*self.m + (1-Beta1)*grad
        self.v = Beta2*self.v + (1-Beta2)*(grad**2)
        self.mBias = Beta1*self.mBias + (1-Beta1)*biasGrad
        self.vBias = Beta2*self.vBias + (1-Beta2)*(biasGrad**2)
        return self.m, self.v, self.mBias, self.vBias
    
    def biasCorrected(self,m1,m2,m1Bias,m2Bias,Beta1=.9,Beta2=.999):
        return m1/(1-Beta1), m2/(1-Beta2),m1Bias/(1-Beta1), m2Bias/(1-Beta2)

    
