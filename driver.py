from NeuralNetwork import NeuralNetwork
from Layer import FullyConnectedLayer
from DataHandler import DataHelper
import pandas as pd
import numpy as np

np.random.seed(100)
df = (pd.read_csv("datasets/breast_cancer.csv")).dropna()
columns = df.columns.to_list()

df.replace({'Class': {2: 0, 4: 1}}, inplace=True)
df = df.to_numpy()
df = df.astype(int)
split = np.split(df, [9], axis=1)

dataframe = split[0]
labels = split[1]

unique_values = np.unique(labels)
num_values = 2
encoded_data = np.zeros((len(labels), num_values), dtype=int)
for i, value in enumerate(labels):
    index = np.where(unique_values == value)[0][0]
    encoded_data[i, index] = 1

labels = encoded_data

print("Cases in dataset: ", len(labels))

trainSet, trainLabels, testSet, testLabels = DataHelper.trainTestSplit(dataframe, labels)

Standardizer = DataHelper.standardizer(trainSet)
trainSet = DataHelper.standardizeCompute(trainSet,Standardizer)
testSet = DataHelper.standardizeCompute(testSet,Standardizer)

print("train set, train labels, test set, train labels shapes:")
print(trainSet.shape)
print(trainLabels.shape)
print(testSet.shape)
print(testLabels.shape)



x = NeuralNetwork(
                  #batchSize=546,
                  hidden1 = FullyConnectedLayer(numNodes=10,activation='ReLU'),
                  hidden2 = FullyConnectedLayer(numNodes=10,activation='ReLU'),
                  hidden3 = FullyConnectedLayer(numNodes=10,activation='ReLU'),
                  hidden4 = FullyConnectedLayer(numNodes=10,activation='ReLU'),
                  output = FullyConnectedLayer(numNodes=2,activation='softmax')
                 )
# specify Adam and AdamW. weight decay means nothing if used with Adam
x.train(trainSet, trainLabels, epochs=1000, eta=0.01,loss='AdamW', weightDecay=.25)


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
