'''
@author: Patrick Erickson
Project Name: NeuralNetwork
Project Description: Constructs a modular implementation of an artificial neural network
'''
import numpy as np
from NeuralNetworkScripts.Layer import FullyConnectedLayer
import h5py

__all__ = [
    '__init__',
    'test',
    'train',
    'adjustBatches',
    'loadModel',
    'evaluate',
    'predict',
    'saveModel'
]


class NeuralNetwork:
    '''
    This class creates a modular neural network that allows you to customize the number of layers. Handles 
    forward pass and backpropagation, as well as the connection of the neural network. FullyConnectedLayers are treated
    as nodes. If model is specified, the 

    FUNCTIONS:
    __init__: constructs the model
    test: takes in a test dataset and its labels and returns the loss and guesses in the same dimensions of returned labels
    train: trains the model on specified params
    adjustBatches: readjusts the batch size for every layer in the network
    saveModel: saves the model as a .h5 file.
    _insertEnd: Helper to chain FullyConnectedLayers together with specified batch size.
    _checkDropOut: handler for dropout masks to multiply the neurons with
    _forwardPassTest: specialized helper for test to handle truncated samples from batching
    _forwardPassTrain: specialized helper for train to handle batches of exactly specified size
    _computeLoss: helper for backpropagation. Produces loss p_i - y_i
    _backPropation: backpropagates network and updates weights based on gradients
    '''

    def __init__(self,model=None, batchSize=32,inputDropout=0,**kwargs):
        '''
        Constructs a Neural Network object with the specified layers as nodes. 
        Utilizes insertEnd to chain nodes together. Does special error handling to
        ensure min args for proper Neural network construction

        INPUTS:
        batchSize: the size of the batch specified by the user. Default 32
        inputDropout: gives the Binomial probability of the input mask. Default 0 (no dropout)
        '''
        if model is None:
            #automatic input layer handling
            self.numLayers = 1
            self.head = FullyConnectedLayer(1,"linear",inputDropout)
            self.tail = self.head


            self.batchSize = batchSize

            # add all specified layers to the dataset
            for key, item in kwargs.items():
                self._insertEnd(item)

            # error handling
            if self.batchSize < 1:
                raise Exception("Enter a valid batchSize.")
            if self.numLayers < 2:
                raise Exception("Must specify at least 1 layer for the neural network.")
            if self.tail.activation[2] not in ['softmax','mse']:
                raise Exception("Specify softmax or mse for the final FullyConnectedLayer.")
            curNode = self.tail.prev
            while curNode is not None:
                if curNode.activation[2] in ['softmax','mse']:
                    raise Exception("softmax or mse should only be used for the final layer.")
                curNode = curNode.prev
        else:
            
            #unload with h5py
            with h5py.File(model + ".h5", "r") as f:
                
                self.batchSize = f.attrs['batchSize']

                # take care of initial head first
                dropoutHead = f['dropout_0'][()]
                activationHead = f['activation_0'][()]

                if isinstance(activationHead, bytes): # strings are bytes. Bruh
                    activationHead = activationHead.decode('utf-8')

                self.numLayers = 1
                self.head = FullyConnectedLayer(1,activationHead,dropoutHead)
                self.tail = self.head

                # Grab every other set of weights, activations, and dropouts
                i = 1
                while f.get(f"activation_{i}") is not None:
                    newActivation = f[f"activation_{i}"][()]
                    if isinstance(newActivation, bytes): 
                        newActivation = newActivation.decode('utf-8')
                    self._insertEnd(FullyConnectedLayer(1,newActivation, f[f"dropout_{i}"][()]))
                    i += 1

                curNode = self.head
                
                #rebuild all of the weights, biases and moment sizes from the previous model.
                i = 0
                while curNode.next is not None:
                    curNode.weights = f[f'weights_{i}'][()]
                    curNode.bias =  f[f'bias_{i}'][()]
                    curNode.m = np.zeros_like(curNode.weights[:,:,0])
                    curNode.v = np.zeros_like(curNode.weights[:,:,0])
                    curNode.mBias = np.zeros_like(curNode.bias[:,:,0])
                    curNode.vBias = np.zeros_like(curNode.bias[:,:,0])
                    curNode = curNode.next
                    i+=1

    
    def adjustBatches(self, newBatchSize):
        '''
        Changes the batch size for the entire dataset. Can
        be called by the client once the object is made. Calls on 
        node's adjust batch size to change specific node batches for weights

        INPUT:
        newBatchSize: client-specified batch size
        '''

        self.batchSize = newBatchSize
        curNode = self.head

        while curNode.weights is not None:

            # calls inner layer batch adjust
            curNode.adjustBatchSize(newBatchSize)
            curNode = curNode.next

        # rexpands via broadcasting outside layer because im lazy and dont wanna change it
        curNode.inputs = np.repeat(curNode.inputs[:,:,0:1], newBatchSize, axis=2)
        curNode.activated = np.repeat(curNode.activated[:,:,0:1], newBatchSize, axis=2)


    def train(self,trainSet, trainLabels,learningRate=.01,loss='Adam',epochs=1000,weightDecay=0):
        '''
        Trains the model based on the specified hyperparameters and updates its weights. Returns its final shuffled training and 
        testing sets.

        INPUT:
        trainSet: The data table, as a numpy array with a 2d matrix of casesxdimensions
        trainLabels: The correspoding labels to the trainSet. 
        NOTE: for softmax, you need categorical 1-hot encoding of the labels
        learningRate: learning rate for the gradient updates. Default = .01
        loss: specify either Adam or AdamW. Switches optimizer based on specification.
        epochs: number of iterations over the entire dataset. Default = 1000
        weightDecay: Decay parameter for AdamW. Default to 0 (Adam)
        
        OUTPUT:
        trainingSetShuffled: returns the final shuffled training set so loss can be calculated by client.
        trainingLabelsShuffled: returns the final shuffled training labels so loss can be calculated by client.
        '''

        # adam/adamW handling
        isAdamW = False
        if not (loss == 'Adam' or loss == 'AdamW'):
            raise Exception("Please specify either Adam or AdamW for loss.")
        isAdamW = loss == 'AdamW'

        # locks into specified batchSize if user specifies new batch size.
        self.adjustBatches(self.batchSize) 

        #turn to tensors for training requirement
        trainSet = trainSet.T[np.newaxis,:,:]
        trainLabels = trainLabels.T[np.newaxis,:,:]

        for epoch in range(epochs):

            loss = 0

            #shuffle tensor
            indices = np.random.permutation(trainSet.shape[2])
            trainSetShuffled = trainSet[:, :, indices]
            trainLabelsShuffled = trainLabels[:, :, indices]

            # split dataset into batches
            n = trainSetShuffled.shape[2]
            refactored = (n // self.batchSize) * self.batchSize 
            trainSetTesseract = trainSetShuffled[:, :, :refactored]
            trainLabelTesseract = trainLabelsShuffled[:, :, :refactored]
            
            # aggregate batch tensors to tesseracts
            numBatches = refactored // self.batchSize
            trainSetTesseract = trainSetTesseract.reshape(trainSetShuffled.shape[0],trainSetShuffled.shape[1],numBatches,self.batchSize)
            trainLabelTesseract = trainLabelTesseract.reshape(trainLabelsShuffled.shape[0],trainLabelsShuffled.shape[1],numBatches,self.batchSize)

            for i in range(trainSetTesseract.shape[2]):

                # follows [1,features,numberOfBatches,batchSize]
                trainBatch = trainSetTesseract[:, :, i, :] 
                labelBatch = trainLabelTesseract[:, :, i, :] 
                
                self.head.inputs = trainBatch # set new shuffled batch as the next inputs
                loss += self._forwardPassTrain(labelBatch)
                self._backPropagation(learningRate,labelBatch,weightDecay,isAdamW)
            
            # calculate loss and print
            loss = loss / trainSetTesseract.shape[2]
            if(epochs < 10):
                print("loss for epoch: " , (epoch+1), ":", loss)
            elif (epoch+1) % (int(epochs/10)) == 0:
                print("loss for epoch: " , epoch+1, ":", loss)

        # NOTE: We do not take care of the left over values from the batches. This is a stylistic choice, as we wish to 
        # speed up computational efficiency by not adjusting the batch size for the last iteration. The intuition
        # arises from the fact that for every dataset S, when performing computations of equivalent difficulties,
        # which can be modeled as some inverse proportional relationship between the size of the datset and the number
        # of epochs, the dataset will incorperate the missing datapoints in the leftover batch

        # we return the shuffled train set and labels in case the client wishes to perform validation
        return trainSetShuffled, trainLabelsShuffled
    

    def evaluate(self,testSet,trueLabels):
        '''
        Tests the model based on the specified test set and labels. Returns its loss and predictions.
        Handles the truncated remainder batch to ensure everything is classified and not cut off.
        Does not shuffle the data, as opposed to train.

        INPUT:
        testSet: The data table test set, as a numpy array with a 2d matrix of casesxdimensions
        testLabels: The correspoding labels to the testSet. 
        NOTE: for softmax, you need categorical 1-hot encoding of the labels
        
        OUTPUT:
        loss: returns the overall loss (based off of missed predictions)
        guesses: returns the final predictions for the given test batch
        '''

        # mutate the datset to work with out tensor implementation
        self.adjustBatches(self.batchSize)
        testSet = testSet.T[np.newaxis,:,:]
        trueLabels = trueLabels.T[np.newaxis,:,:]
        guesses = []

        # unlike train, no shuffling
        # split dataset into batches
        n = testSet.shape[2]
        refactored = (n // self.batchSize) * self.batchSize 
        testSetTesseract = testSet[:, :, :refactored]
        testLabelTesseract = trueLabels[:, :, :refactored]
        
        # aggregate batch tensors to tesseracts
        numBatches = refactored // self.batchSize
        testSetTesseract = testSetTesseract.reshape(testSet.shape[0],testSet.shape[1],numBatches,self.batchSize)
        testLabelTesseract = testLabelTesseract.reshape(trueLabels.shape[0],trueLabels.shape[1],numBatches,self.batchSize)
        self.head.inputs = testSet

        loss = 0
        for i in range(testSetTesseract.shape[2]):

            # follows [1,features,numberOfBatches,batchSize]
            testBatch = testSetTesseract[:, :, i, :]   
            labelBatch = testLabelTesseract[:, :, i, :]  

            # specialized forward pass for each batch
            self.head.inputs = testBatch
            loss += self._forwardPassTest(labelBatch)
            guesses.append(self.tail.activated[0].T)

        # there are ramainders per batch that are dropped during training
        # we need to handle them here for consistency
        remainder = n % self.batchSize
        numBatches = testSetTesseract.shape[2]
        if remainder != 0:

            testBatch = testSet[:, :, refactored:]  # remaining samples
            labelBatch = trueLabels[:, :, refactored:]
            self.adjustBatches(testBatch.shape[2])
            self.head.inputs = testBatch
            loss += self._forwardPassTest(labelBatch)
            self.adjustBatches(self.batchSize)
            guesses.append(self.tail.activated[0].T)
            numBatches += 1 

        # return the test loss and the batches
        guesses = np.concatenate(guesses, axis=0)
        loss = loss / numBatches
        return loss, guesses

    def predict(self,testSet):
        '''
        classifies the items of the specified set. Returns predictions. Does not require labels for loss.
        Handles the truncated remainder batch to ensure everything is classified and not cut off.
        Does not shuffle the data, as opposed to train.

        INPUT:
        testSet: The data table test set, as a numpy array with a 2d matrix of casesxdimensions
        NOTE: for softmax, you need categorical 1-hot encoding of the labels
        
        OUTPUT:
        guesses: returns the final predictions for the given test batch
        '''

        # mutate the datset to work with out tensor implementation
        self.adjustBatches(self.batchSize)
        testSet = testSet.T[np.newaxis,:,:]
        guesses = []

        # unlike train, no shuffling
        # split dataset into batches
        n = testSet.shape[2]
        refactored = (n // self.batchSize) * self.batchSize 
        testSetTesseract = testSet[:, :, :refactored]
        
        # aggregate batch tensors to tesseracts
        numBatches = refactored // self.batchSize
        testSetTesseract = testSetTesseract.reshape(testSet.shape[0],testSet.shape[1],numBatches,self.batchSize)
        self.head.inputs = testSet
        
        for i in range(testSetTesseract.shape[2]):

            # follows [1,features,numberOfBatches,batchSize]
            testBatch = testSetTesseract[:, :, i, :]   

            # specialized forward pass for each batch
            self.head.inputs = testBatch
            self._forwardPassTest()
            guesses.append(self.tail.activated[0].T)

        # there are ramainders per batch that are dropped during training
        # we need to handle them here for consistency
        remainder = n % self.batchSize
        numBatches = testSetTesseract.shape[2]
        if remainder != 0:

            testBatch = testSet[:, :, refactored:]  # remaining samples
            self.adjustBatches(testBatch.shape[2])
            self.head.inputs = testBatch
            self._forwardPassTest()
            self.adjustBatches(self.batchSize)
            guesses.append(self.tail.activated[0].T)
            numBatches += 1 

        # return the test loss and the batches
        guesses = np.concatenate(guesses, axis=0)
        return guesses


    def saveModel(self, fileName: str):
        '''
        Saves a model specified by the fileName parameter. Can be reaccessed by creating a new instance of 
        NeuralNetwork. ie. savedModel = NeuralNetwork(model=fileName). This should not include the .h5 extension.

        INPUT:
        fileName: the name of the model to be created. Excludes .h5 extension.
        '''
        curNode = self.head

        i = 0 

        # we save only weights, layers, biases, activations, dropouts, and the batchSize
        with h5py.File(fileName + ".h5", "w") as f:
            while curNode.next is not None:

                if curNode.weights is not None:
                    f.create_dataset(f"weights_{i}", data=curNode.weights)
                if curNode.bias is not None:
                    f.create_dataset(f"bias_{i}", data=curNode.bias)
                # Lambdas are tricky, so we opted to have an identifier string
                # saved and reconstructed. this is activation[2].
                f.create_dataset(f"activation_{i}", data=curNode.activation[2])
                f.create_dataset(f"dropout_{i}", data=curNode.dropout)
                curNode = curNode.next
                i += 1
            f.create_dataset(f"activation_{i}", data=curNode.activation[2])
            f.create_dataset(f"dropout_{i}", data=curNode.dropout)

            f.attrs['batchSize'] = self.batchSize

    def _insertEnd(self, toAdd):
        '''
        Chains fully connected layers together. Performs a lot of handling and initializations due to the fact that
        batch size is intialized in the higher priority class of NeuralNetwork (not FullyConnected Layer). Broadcasts
        the correct dimensions to weight matrices and also initializes momentum and moving average of momentum to this
        broadcast as well. Weights are initilialized via He.

        INPUT:
        toAdd: the FullyConnected Layer to connect.
        '''

        if not isinstance(toAdd, FullyConnectedLayer):
            raise Exception("Error when inserting to Neural Network: Must pass in a FullyConnectedLayer object")
        
        # use He scaling NOTE: Super important for model convergence
        fan_in = self.tail.featureSize
        scale = np.sqrt(2 / fan_in)

        self.tail.weights = np.random.randn(toAdd.featureSize, self.tail.featureSize, 1) * scale
        # no need for He scaling for bias
        self.tail.bias = np.zeros((1, toAdd.featureSize, 1))

        toAdd.inputs = np.zeros((1,toAdd.featureSize,1))
        toAdd.activated = np.zeros((1,toAdd.featureSize,1))

        # for adam momentum (m) and momentum moving average
        self.tail.m = np.zeros_like(self.tail.weights[:,:,0])
        self.tail.v = np.zeros_like(self.tail.weights[:,:,0])
        self.tail.mBias = np.zeros_like(self.tail.bias[:,:,0])
        self.tail.vBias = np.zeros_like(self.tail.bias[:,:,0])

        #connect the tail to the network
        self.tail.next = toAdd
        toAdd.prev = self.tail
        self.tail = toAdd

        # forces batches to be same size, since layers are initialized with a default batchsize
        self.adjustBatches(self.batchSize)
        self.numLayers += 1

    
    def _checkDropOut(self, toCheck, input=False):
        '''
        sets a mask for any given layer with a binomial distribution
        (sum of bernoulli) to multiply the weight matrix for the specific layer by,
        effectively "zeroing" random nodes.

        INPUT: 
        toCheck: the node to check if dropout exists or not
        input: boolean var on if we are performing input dropout or not. Default is false.

        OUTPUT: binomial 1/0 generated mask with specified probability. Used to set a nodes mask.
        '''

        size = toCheck.weights.shape[0]

        # if input is specified, we make an input mask instead
        if input:
            size = toCheck.inputs.shape[1] 
        mask = [1]
        if toCheck.dropout != 0:
            mask = (1/(1-toCheck.dropout))*np.random.binomial(1,(1-toCheck.dropout),size)

            # edge case where the dropout is so extreme, all nodes turn off and the model essentially "disconnects"
            if len(np.unique(mask)) == 1:
                toCheck.mask[0] = 1/(1-toCheck.dropout)
        return mask


    def _forwardPassTest(self, labels=None):
        '''
        A simpler forward pass, as we don't need to apply dropout masks

        INPUT:
        labels: the test labels, transformed into tensor format

        OUTPUT:
        loss: returns loss for the forward pass
        '''

        curNode = self.head

        while curNode.next is not None:

            # performs activation for nonlinearity from the previous layer's output (this layer's input)
            curNode.activated = curNode.activation[0](curNode.inputs)

            # multiplies WTx + b, where x is the activated from the previous layer, multiplied by weights and biases, and sets this to 
            # the input of the next layer
            # einsum contains a transpose for W and matmul and sums along column dimension
            curNode.next.inputs = np.einsum('ijk,ljk->ilk', curNode.activated, curNode.weights) + curNode.bias
            curNode = curNode.next
        curNode.activated = curNode.activation[0](curNode.inputs)

        # return loss for the entire forward pass
        if labels is None:
            return curNode.activation[0](curNode.activated)
        else:
            return curNode.activation[3](curNode.activated, labels)

    def _forwardPassTrain(self, labels=None):
        '''
        Forward pass with dropout implementation. Handles masking of weights and/or inputs

        INPUT:
        labels: the train labels, transformed into tensor format.

        OUTPUT:
        loss: returns loss for the forward pass
        '''

        curNode = self.head # handles input dropout

        # handles input dropout and comput the first activation a_{i,j}^{(l)}
        curNode.activated = curNode.activation[0](curNode.inputs)
        curNode.mask = self._checkDropOut(curNode, input=True)

        #multiplies the input by the binomially generated dropout mask
        inputsWithDropout = np.einsum('j,ijk->ijk',curNode.mask,curNode.activated)
        curNode.mask = self._checkDropOut(curNode)

        # multiplies the weight mask by the weight matrix. This essentially "zeros" the node, and doesnt allow it to contribute for
        # that iteration
        weightsWithDropout = np.einsum('i,ijk->ijk',curNode.mask,curNode.weights)

        # multiplies WTx + b, where x is the activated from the previous layer, multiplied by weights and biases, and sets this to 
        # the input of the next layer
        # einsum contains a transpose for W and matmul and sums along column dimension
        curNode.next.inputs = np.einsum('ijk,ljk->ilk', inputsWithDropout, weightsWithDropout) + curNode.bias
        curNode = curNode.next

        while curNode.next is not None:
            # forward pass computations with dropout masks, specified per layer
            curNode.activated = curNode.activation[0](curNode.inputs)
            curNode.mask = self._checkDropOut(curNode)

            # multiplies the mask by the weight matrix. This essentially "zeros" the node, and doesnt allow it to contribute for
            # that iteration
            # NOTE: This is the same as outside the loop
            weightsWithDropout = np.einsum('i,ijk->ijk',curNode.mask,curNode.weights)

            # multiplies WTx + b, where x is the activated from the previous layer, multiplied by weights and biases, and sets this to 
            # the input of the next layer
            # einsum contains a transpose for W and matmul and sums along column dimension
            # NOTE: This is the same as outside the loop
            curNode.next.inputs = np.einsum('ijk,ljk->ilk', curNode.activated, weightsWithDropout) + curNode.bias 
            curNode = curNode.next
        curNode.activated = curNode.activation[0](curNode.inputs)

        # return loss for the entire forward pass
        if labels is None:
            return curNode.activation[0](curNode.activated)
        else:
            return curNode.activation[3](curNode.activated, labels)

    
    def _computeLoss(self, trueLabels):
        '''
        Helper function for backpropagation. Returns loss to be minimized, not reported
        loss.

        INPUT:
        trueLabels: the labels corresponding the the trainSet

        OUTPUT:
        delta: returns the derivative of either MSE or softmax
        '''

        curNode = self.tail
        return curNode.activation[1](curNode.activated,trueLabels)

    def _backPropagation(self, eta, trueLabels, lam, isAdamW):
        '''
        Backpropagates through the network during training and updates the weights based on the given hyperparameters

        NOTE: delta(l) is the loss for the layer weight respect to the 
        weights of all the layers after it, delta(l-1) is loss for previous layer. Any (l) represents
        the current layer
        NOTE: z(l) is the input for that layer
        NOTE: a(l) is the activated input for that layer
        NOTE: w(l) is the weight matrix for that layer
        NOTE: because of how we build our matrix and go through backpropagation, the 
        previous inputs are within the current node when examined. This explains why the 
        formulas may not follow the exact mathematical formulations.

        INPUTS:
        eta: learning Rate. Specified in train
        trueLabels: the labels from the trainSet, tensorfied for our architecture.
        lam: adamW regularization parameter
        isAdamW: boolean operator determining if AdamW needs to be calculated. Otherwise perform 0 multiplication
        for reduction of computational overhead.

        NOTE: Weights are auto updated, and nothing is returned.
        '''

        curNode = self.tail.prev
        # compute delta(l)*a(l-1) for gradient of weight block 3
        deltaLoss = self._computeLoss(trueLabels)
        weightsPerLayer = 0
        while curNode is not None:
            
            #delta (aka. loss) w/r to the layer, multiplied by mask, if specified
            weightsWithDropout = np.einsum('i,ijk->ijk',curNode.mask,curNode.weights)

            # we do not compute adamW weigh matrix for faster computation when adam is specified
            if(isAdamW):
                weightsPerLayer = weightsWithDropout[:,:,0]

            # calculate new gradients delta(l)*a(l)T
            # einsum takes care of the a(l)T broadcasting and transpose
            gradient = np.einsum('ijk,jlk->jlk',deltaLoss,curNode.activated)
            # bias is just the loss for every layer, aka delta(l) 
            biasGradient = deltaLoss 
            # compute new loss delta for previous hidden layer
            wTDeltaLoss = np.einsum('ijk,lik->ljk',(weightsWithDropout),deltaLoss)

            # get the derivative w/r to the activation of the previous layer f'(z(l-1))
            # NOTE: this is z(l-1) because each Layer node saves the previous layer's outputs
            activationDerivative = curNode.activation[1](curNode.inputs)

            # compute delta(l-1) via hadamard product
            deltaLoss = np.multiply(wTDeltaLoss,activationDerivative)

            #average w/r to the size of the batch
            gradient = np.mean(gradient, axis=2)
            biasGradient = np.mean(biasGradient, axis=2)

            # calculating moments and moving averages per layer for Adam
            m,v,mBias,vBias = curNode.newMoment(gradient,biasGradient)
            mHat,vHat,mHatBias,vHatBias = curNode.biasCorrected(m,v,mBias,vBias)

            # lam and/or weightsPerLayer is set o 0 for adam, some positive number O.W. for AdamW
            etaGrad = eta*(mHat / (np.sqrt(vHat) + 1e-8) + lam*weightsPerLayer) 
            etaGradBias = eta*(mHatBias / (np.sqrt(vHatBias) + 1e-8))

            # gradient descent
            weightsToUpdate = curNode.weights[:,:,0] - etaGrad
            biasToUpdate = curNode.bias[:,:,0] - etaGradBias

            #re-expand and update weights and biases for the update
            curNode.weights = np.repeat(weightsToUpdate[:,:,np.newaxis], self.batchSize, axis=2)
            curNode.bias = np.repeat(biasToUpdate[:,:,np.newaxis], self.batchSize, axis=2)
            curNode = curNode.prev