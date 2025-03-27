from NeuralNetwork import NeuralNetwork
from Layer import FullyConnectedLayer
from DataHandler import DataHelper
import numpy as np
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical 


# Load the MNIST dataset
(digitsTrainSet, digitsTrainLabels), (digitsTestSet, digitsTestLabels) = mnist.load_data()
trainLabels = to_categorical(digitsTrainLabels, num_classes=10)
testLabels = to_categorical(digitsTestLabels, num_classes=10)
trainSet = digitsTrainSet.reshape(60000, 28*28) # stretch out to naively get 784 features
testSet = digitsTestSet.reshape(10000, 28*28)

#trainSet, trainLabels, testSet, testLabels = DataHelper.trainTestSplit(trainSet, trainLabels) # cross-validation used to find hyperparameters
np.random.seed(100) # unseeded for cv


Standardizer = DataHelper.standardizer(trainSet)
trainSet = DataHelper.standardizeCompute(trainSet,Standardizer)
testSet = DataHelper.standardizeCompute(testSet,Standardizer)

print("train set, train labels, test set, train labels shapes:")
print(trainSet.shape)
print(trainLabels.shape)
print(testSet.shape)
print(testLabels.shape)


x = NeuralNetwork(
                  # NOTE: Example params
                  #batchSize=128, #default 32
                  #inputDropout=.2,
                  hidden2 = FullyConnectedLayer(numNodes=400,activation='ReLU',dropout=.25), 
                  hidden3 = FullyConnectedLayer(numNodes=400,activation='ReLU',dropout=.25),
                  output = FullyConnectedLayer(numNodes=10,activation='softmax')
                 )

# specify Adam and AdamW. weight decay means nothing if used with Adam
x.train(trainSet, trainLabels, epochs=12, eta=.001,loss='AdamW',weightDecay=.15) # best epochs=10
lossTraining, trainGuesses = x.test(trainSet, trainLabels)

predicted_train = np.argmax(trainGuesses, axis=1)
true_train = np.argmax(trainLabels, axis=1)
train_accuracy = np.mean(predicted_train == true_train) * 100

print("\nFinal Training Loss:", lossTraining)
print("Training Acc: {:.2f}%".format(train_accuracy))

lossTesting, testGuesses = x.test(testSet, testLabels)

predicted_test = np.argmax(testGuesses, axis=1)
true_test = np.argmax(testLabels, axis=1)
test_accuracy = np.mean(predicted_test == true_test) * 100

print("\nFinal Testing Loss:", lossTesting)
print("Testing Acc: {:.2f}%".format(test_accuracy))
