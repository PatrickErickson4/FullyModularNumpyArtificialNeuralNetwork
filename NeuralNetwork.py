import numpy as np
from Layer import FullyConnectedLayer

class NeuralNetwork:
    
    def __init__(self,inputs,batchSize=1,**kwargs):

        #input layer
        self.head = FullyConnectedLayer(inputs,"linear",first=True)
        self.tail = self.head
        self.batchSize = batchSize

        for key, item in kwargs.items():
            self.insertEnd(item)



    def insertEnd(self, toAdd):

        if not isinstance(toAdd, FullyConnectedLayer):
            raise Exception("Error when inserting to Layer LinkedList: Must pass in a FullyConnectedLayer object")
        
        # use He scaling
        fan_in = self.tail.featureSize
        scale = np.sqrt(2 / fan_in)

        weightsToExpand = np.random.randn(toAdd.featureSize, self.tail.featureSize, 1) * scale
        self.tail.weights = np.repeat(weightsToExpand, self.batchSize, axis=2)

        # For biases, it's common to initialize to zero (or small constant)
        biasToExpand = np.zeros((1, toAdd.featureSize, 1))
        self.tail.bias = np.repeat(biasToExpand, self.batchSize, axis=2)

        # for adam
        self.tail.m = np.zeros_like(self.tail.weights[:,:,0])
        self.tail.v = np.zeros_like(self.tail.weights[:,:,0])

        self.tail.mBias = np.zeros_like(self.tail.bias[:,:,0])
        self.tail.vBias = np.zeros_like(self.tail.bias[:,:,0])

        self.tail.next = toAdd
        toAdd.prev = self.tail
        self.tail = toAdd
    
    def adjustBatches(self, newBatchSize):
        self.batchSize = newBatchSize
        curNode = self.head
        while curNode.weights is not None:
            curNode.adjustBatchSize(newBatchSize)
            curNode = curNode.next
        curNode.inputs = np.repeat(curNode.inputs[:,:,0:1], newBatchSize, axis=2)
        curNode.activated = np.repeat(curNode.activated[:,:,0:1], newBatchSize, axis=2)

    def forwardPass(self):
        curNode = self.head
        
        while curNode.next is not None:
            curNode.activated = curNode.activation[0](curNode.inputs)
            curNode.next.inputs = np.einsum('ijk,ljk->ilk', curNode.activated, curNode.weights) + curNode.bias
            curNode = curNode.next
        curNode.activated = curNode.activation[0](curNode.inputs)

        return curNode.activated


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
        print("Tail activated and input dims:",self.tail.activated.shape)
        print(self.tail.inputs.shape)
            
    def computeLoss(self, trueLabels):
        curNode = self.tail
        return curNode.activation[1](curNode.activated,trueLabels)

    def backPropagation(self, eta, trueLabels):

        curNode = self.tail.prev
        # compute delta(l)*a(l-1) for gradient of weight block 3
        gammaLoss = self.computeLoss(trueLabels)
        
        while curNode is not None:

            gradient = np.einsum('ijk,jlk->jlk',gammaLoss,curNode.activated)
            biasGradient = gammaLoss # bias is just the loss for every layer
            # compute new loss gamma2 for previous hidden layer
            wTgammaLoss = np.einsum('ijk,lik->ljk',curNode.weights,gammaLoss)

            activationDerivative = curNode.activation[1](curNode.inputs)
            gammaLoss = np.multiply(wTgammaLoss,activationDerivative)

            #average w/r to the size of the batch
            gradient = np.mean(gradient, axis=2)
            biasGradient = np.mean(biasGradient, axis=2)

            m,v,mBias,vBias = curNode.newMoment(gradient,biasGradient)
            mHat,vHat,mHatBias,vHatBias = curNode.biasCorrected(m,v,mBias,vBias)

            etaGrad = eta*(mHat / (np.sqrt(vHat) + 1e-8))
            etaGradBias = eta*(mHatBias / (np.sqrt(vHatBias) + 1e-8))

            weightsToUpdate = curNode.weights[:,:,0] - etaGrad
            biasToUpdate = curNode.bias[:,:,0] - etaGradBias

            #re-expand and update weights and biases for the update
            curNode.weights = np.repeat(weightsToUpdate[:,:,np.newaxis], self.batchSize, axis=2)
            curNode.bias = np.repeat(biasToUpdate[:,:,np.newaxis], self.batchSize, axis=2)
            curNode = curNode.prev
    



    # add batch splitter, train sizes, etx
    def train(self,trainSet, trainLabels,eta=.01,epochs=1000):

        for epoch in range(epochs):

            # for batch in batchSize
            indices = np.random.permutation(trainSet.shape[2])
            trainSetShuffled = trainSet[:, :, indices]
            trainLabelsShuffled = trainLabels[:, :, indices]
            self.head.inputs = trainSetShuffled

            self.forwardPass()
            if epoch % (int(epochs/10)) == 0:
                print("loss for epoch: " , epoch, ":", self.computeLoss(trainLabelsShuffled))
            self.backPropagation(eta, trueLabels=trainLabelsShuffled)






