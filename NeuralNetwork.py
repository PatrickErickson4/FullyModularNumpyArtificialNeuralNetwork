import numpy as np
from Layer import FullyConnectedLayer

class NeuralNetwork:
    
    def __init__(self,batchSize=32,inputDropout=0,**kwargs):

        #input layer
        self.numLayers = 1
        self.head = FullyConnectedLayer(1,"linear",inputDropout)
        self.tail = self.head
        self.batchSize = batchSize

        for key, item in kwargs.items():
            self.insertEnd(item)
        if self.numLayers < 2:
            raise Exception("Must specify at least 1 layer for the neural network.")

    def insertEnd(self, toAdd):

        if not isinstance(toAdd, FullyConnectedLayer):
            raise Exception("Error when inserting to Layer LinkedList: Must pass in a FullyConnectedLayer object")
        
        # use He scaling NOTE: Super important for model convergence
        fan_in = self.tail.featureSize
        scale = np.sqrt(2 / fan_in)

        self.tail.weights = np.random.randn(toAdd.featureSize, self.tail.featureSize, 1) * scale
        # For biases, it's common to initialize to zero (or small constant)
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

        # i know this sucks, but i dont care.
        self.adjustBatches(self.batchSize)
        self.numLayers += 1

    
    def adjustBatches(self, newBatchSize):
        self.batchSize = newBatchSize
        curNode = self.head
        while curNode.weights is not None:
            curNode.adjustBatchSize(newBatchSize)
            curNode = curNode.next
        curNode.inputs = np.repeat(curNode.inputs[:,:,0:1], newBatchSize, axis=2)
        curNode.activated = np.repeat(curNode.activated[:,:,0:1], newBatchSize, axis=2)

    def checkDropOut(self, toCheck, input=False):
        size = toCheck.weights.shape[0]
        if input:
            size = toCheck.inputs.shape[1]
        mask = [1]
        if toCheck.dropout != 0:
            mask = (1/(1-toCheck.dropout))*np.random.binomial(1,(1-toCheck.dropout),size)
            if len(np.unique(mask)) == 1:
                toCheck.mask[0] = 1/(1-toCheck.dropout)
        return mask
    
    def test(self,testSet,trueLabels):
        
        self.adjustBatches(len(testSet))
        testSet = testSet.T[np.newaxis,:,:]
        trueLabels = trueLabels.T[np.newaxis,:,:]

        self.head.inputs = testSet
        loss = self.forwardPassTest(trueLabels)

        return loss, self.tail.activated[0].T


    def forwardPassTest(self, labels):

        curNode = self.head

        while curNode.next is not None:
            curNode.activated = curNode.activation[0](curNode.inputs)
            curNode.next.inputs = np.einsum('ijk,ljk->ilk', curNode.activated, curNode.weights) + curNode.bias
            curNode = curNode.next
        curNode.activated = curNode.activation[0](curNode.inputs)

        return curNode.activation[3](curNode.activated, labels)

    def forwardPassTrain(self, labels):

        curNode = self.head # handles input dropout

        curNode.activated = curNode.activation[0](curNode.inputs)
        curNode.mask = self.checkDropOut(curNode, input=True)
        inputsWithDropout = np.einsum('j,ijk->ijk',curNode.mask,curNode.activated)
        curNode.mask = self.checkDropOut(curNode)
        weightsWithDropout = np.einsum('i,ijk->ijk',curNode.mask,curNode.weights)
        curNode.next.inputs = np.einsum('ijk,ljk->ilk', inputsWithDropout, weightsWithDropout) + curNode.bias
        curNode = curNode.next

        while curNode.next is not None:
            curNode.activated = curNode.activation[0](curNode.inputs)
            curNode.mask = self.checkDropOut(curNode)
            weightsWithDropout = np.einsum('i,ijk->ijk',curNode.mask,curNode.weights)
            curNode.next.inputs = np.einsum('ijk,ljk->ilk', curNode.activated, weightsWithDropout) + curNode.bias
            curNode = curNode.next
        curNode.activated = curNode.activation[0](curNode.inputs)

        return curNode.activation[3](curNode.activated, labels)
    
    '''
    # used for debugging
    def viewNetwork(self):
        curNode = self.head
        counter = 1
        while curNode.next is not None:
            print(f"Layer {counter}:")
            print("bias shape: ", curNode.bias.shape)
            print("bias: ", curNode.bias)
            print("weight shape: ", curNode.weights.shape)
            print("weights: ", curNode.weights)
            print("Input and activated dims:",curNode.activated.shape)
            print(curNode.inputs.shape)
            counter += 1
            curNode = curNode.next
        print("Tail activated and input dims:", self.tail.activated.shape)
        print(self.tail.inputs.shape)
    '''

    def computeLoss(self, trueLabels):
        curNode = self.tail
        return curNode.activation[1](curNode.activated,trueLabels)

    def backPropagation(self, eta, trueLabels, lam, isAdamW):

        curNode = self.tail.prev
        # compute delta(l)*a(l-1) for gradient of weight block 3
        gammaLoss = self.computeLoss(trueLabels)
        weightsPerLayer = 0
        while curNode is not None:

            weightsWithDropout = np.einsum('i,ijk->ijk',curNode.mask,curNode.weights)
            if(isAdamW):
                weightsPerLayer = weightsWithDropout[:,:,0]
            gradient = np.einsum('ijk,jlk->jlk',gammaLoss,curNode.activated)
            biasGradient = gammaLoss # bias is just the loss for every layer
            # compute new loss gamma2 for previous hidden layer
            wTgammaLoss = np.einsum('ijk,lik->ljk',(weightsWithDropout),gammaLoss)

            activationDerivative = curNode.activation[1](curNode.inputs)
            gammaLoss = np.multiply(wTgammaLoss,activationDerivative)

            #average w/r to the size of the batch
            gradient = np.mean(gradient, axis=2)
            biasGradient = np.mean(biasGradient, axis=2)

            m,v,mBias,vBias = curNode.newMoment(gradient,biasGradient)
            mHat,vHat,mHatBias,vHatBias = curNode.biasCorrected(m,v,mBias,vBias)

            etaGrad = eta*(mHat / (np.sqrt(vHat) + 1e-8) + lam*weightsPerLayer)
            etaGradBias = eta*(mHatBias / (np.sqrt(vHatBias) + 1e-8))

            weightsToUpdate = curNode.weights[:,:,0] - etaGrad
            biasToUpdate = curNode.bias[:,:,0] - etaGradBias

            #re-expand and update weights and biases for the update
            curNode.weights = np.repeat(weightsToUpdate[:,:,np.newaxis], self.batchSize, axis=2)
            curNode.bias = np.repeat(biasToUpdate[:,:,np.newaxis], self.batchSize, axis=2)
            curNode = curNode.prev
    
    # add batch splitter, train sizes, etx
    def train(self,trainSet, trainLabels,eta=.01,loss='Adam',epochs=1000,weightDecay=0):

        isAdamW = False
        if not (loss == 'Adam' or loss == 'AdamW'):
            raise Exception("Please specify either Adam or AdamW for loss.")
        isAdamW = loss == 'AdamW'

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
                trainBatch = trainSetTesseract[:, :, i, :]   # selects the i-th batch from trainSetTesseract
                labelBatch = trainLabelTesseract[:, :, i, :]   # selects the i-th batch from trainLabelTesseract
                
                self.head.inputs = trainBatch
                loss += self.forwardPassTrain(labelBatch)
                self.backPropagation(eta,labelBatch,weightDecay,isAdamW)
            
            loss = loss / trainSetTesseract.shape[2]
            if(epochs < 10):
                print("loss for epoch: " , (epoch+1), ":", loss)
            elif (epoch+1) % (int(epochs/10)) == 0:
                print("loss for epoch: " , epoch+1, ":", loss)

        return trainSetShuffled, trainLabelsShuffled