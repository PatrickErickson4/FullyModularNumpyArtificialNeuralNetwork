from NeuralNetwork import NeuralNetwork
from Layer import FullyConnectedLayer
import numpy as np
batch = 4

#np.random.seed(100)
trueLabels = np.array([[[0, 1, 1, 0],
                         [1, 0, 0, 1]]])
#print(trueLabels.shape)

input = np.array([[[0, 0, 1, 1],   # first feature for each sample
                       [0, 1, 0, 1]]]) # second feature for each sample

singleTest = trueLabels[0:1,0:2,0:1] 

singleTrain = input[0:1,0:2,0:1]
print("single train", singleTrain)
print("single test", singleTest)
print(singleTest.shape)

print(trueLabels)
#print("input: ", input)

x = NeuralNetwork(
                  inputs = input,
                  hidden1 = FullyConnectedLayer(numNodes=5,activation='ReLU'),
                  hidden2 = FullyConnectedLayer(numNodes=1,activation='linear'),
                  output = FullyConnectedLayer(numNodes=2,activation='softmax')
                 )

print("before:")
#x.viewNetwork()
#print("Forward pass, softmaxed: ",yo )
#print(x.tail.activated.shape)
#print(trueLabels.shape)

#x.forwardPass()

#print(singleTest)


x.train(input,trueLabels)


print("after: ")

x.head.inputs = singleTrain
x.forwardPass()
print("0 1 for XOR: ", singleTest)
guess = np.mean(x.tail.activated, axis=2)
print("Guess:" , guess)

x.viewNetwork()
'''

x.adjustBatches(3)
x.train(input,trueLabels)
print("after: ")

x.head.inputs = singleTrain
x.forwardPass()
print("0 1 for XOR: ", singleTest)
guess = np.mean(x.tail.activated, axis=2)
print("Guess:" , guess)

print("First viewnet")
x.viewNetwork()
'''
x.adjustBatches(4)
#x.train(input,trueLabels)

x.viewNetwork()



print("after: ")

x.head.inputs = singleTrain
x.forwardPass()
print("0 1 for XOR: ", singleTest)
guess = np.mean(x.tail.activated, axis=2)
print("Guess:" , guess)


'''
print("Second viewnet")
x.viewNetwork()

x.adjustBatches(10)
x.viewNetwork()

x.adjustBatches(3)
x.viewNetwork()s
'''