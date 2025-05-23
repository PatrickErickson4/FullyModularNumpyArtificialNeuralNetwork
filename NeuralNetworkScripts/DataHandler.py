'''
@author: Patrick Erickson
Project Name: Data Handler
Project Description: Constructs a modular implementation of useful data manipulation methods, inspired
by sci-kit learn
'''
import numpy as np

class DataHelper:
    '''
    This class contains useful functions for training, splitting, standardizing, and un-normalizing the data.
    NOTE: This only works with 2D matrices/data tables in numpy (not pandas compatible)
    '''
    
    def trainTestSplit(features, labels, trainSize=None, testSize=None, randomState=None):
        '''
        Used on a split label and feature dataset. Will automatically handle single input training and testing.
        Ensures a random shuffling of the train/test. Client can specify train size, test size, or both.
        If no train size is stated, automatically does a randomized 80/20 train/test split.

        INPUT: 
        features: the feature datset, labels removed
        labels: the labels corresponding the the feature dataset
        trainSize: allocated percentage of datset to be meant for training, represented as decimal.
        testSize: allocated percentage of datset to be meant for testing, represented as decimal.
        randomState: Can be specified by client. Will use an int to see the train-test split for reproduceability.

        OUTPUT:
        trainFeatures: the training set split as requested
        testFeatures: the training labels split as requested
        testFeatures: the testing set split as requested
        testLabels: the testing labels split as requested
        '''

        cases = len(features)

        # if else ladders to handle bad input and to correctly set train and test size based on user input.
        if randomState is not None:
            np.random.seed(randomState)

        if cases < 2:
            raise Exception("Your dataset is too small.")

        if trainSize is None and testSize is None:
            trainSize, testSize = 0.8, 0.2
        if trainSize is not None and testSize is None:
            if trainSize <= 0.0 or trainSize >= 1.0:
                raise Exception("Bad Train/Test Split")
            testSize = 1 - trainSize
        elif testSize is not None and trainSize is None:
            if testSize <= 0.0 or testSize >= 1.0:
                raise Exception("Bad Train/Test Split")
            trainSize = 1 - testSize
        #need to handle floating point errors.
        if trainSize is not None and testSize is not None:
            if abs(trainSize + testSize - 1.0) > 1e-6 or trainSize < 0.0 or trainSize > 1.0 or testSize < 0.0 or testSize > 1.0:
                raise Exception("Bad Train/Test Split")

        #randomly shuffles indices of size of feature rows, then partitions the array into a train and
        #test indices
        splitIndex = np.random.permutation(cases)
        lastTrainIndex = int(cases*trainSize)
        trainIndices = splitIndex[:lastTrainIndex]
        testIndices = splitIndex[lastTrainIndex:]

        #indices are used to reshuffle the train and test splits.
        trainFeatures = features[trainIndices]
        trainLabels = labels[trainIndices]
        testFeatures = features[testIndices]
        testLabels = labels[testIndices]

        return trainFeatures, trainLabels, testFeatures, testLabels 
    
    def toOneHot(labels, order=None):
        """
        Takes in an array or 1xn matrix of testing labels and
        converts it into its respective categorical one-hot representation.
        If no order is specified, we manually search the entire label set to build our own one-hot encoding.

        INPUT:
        labels: the labels to one-hot encode
        order: a list of categories, in the specific order specified by the client

        OUTPUT:
        toRet: output of one-hot encoded labels as a (row,categories) numpy matrix
        mapping: a dictionary of the categorical order of the one-hot labels, based on the
        names/numbers in the categorical variable.
        """
        arr = np.array(labels).reshape(-1)
        
        # if an ordered array of categories is specified
        if order is not None:
            order = list(order)
            mapping = {key: i for i, key in enumerate(order)}
        # O.W.
        else:
            uniqueValues = sorted(set(arr))
            mapping = {value: i for i, value in enumerate(uniqueValues)}
        
        num_classes = len(mapping)
        toRet = np.zeros((len(arr), num_classes))
        for i, item in enumerate(arr):
            toRet[i, mapping[item]] = 1
        
        # return the dictionary of the categorical orders if need be.
        return toRet, mapping 

    def standardizer(trainingDataset):
        '''
        Used to ensure that the training set and testing set will have a mean of 0 and
        standard deviation of one. This allows for better convergence.

        NOTE: use the standardizer on ONLY the training set, then use
        StandardizeCompute on both the training and testing set to avoid data leakage

        INPUT:
        trainingDataset: The training set, labels removed.

        OUTPUT:
        an array of:
            mean: means of every feature in the dataset
            std: the standard deviations of every feature in the dataset
        '''
        mean = np.mean(trainingDataset, axis=0)
        std = np.std(trainingDataset, axis=0)
        std[std == 0] = 1.0
        return [mean, std]
    
    def standardizeCompute(toStandardize,standardizer):
        '''
        function used to ensure standardization of all of the features
        to minimize scaling differences and better convergence. to be called as such:

        standardizer = DataHelper.standardizer(trainingSet)
        standardizedTrainingSet = DataHelper.standardizeCompute(trainingSet,standardizer)
        standardizedTestingSet  = DataHelper.standardizeCompute(trainingSet,standardizer)

        NOTE: use the standardizer on ONLY the training set, then use
        StandardizeCompute on both the training and testing set to avoid data leakage

        INPUT:
        trainingDataset: The training set, labels removed.
        standardizer: an array of:
            mean: means of every feature in the dataset
            std: the standard deviations of every feature in the dataset

        OUTPUT:
        standardizedSet: the resulting dataset, standardized on the training dataset.
        '''
        return (toStandardize-standardizer[0]) / standardizer[1]
    
    
    def unNormalize(tounNormalize, standardizer):
        '''
        function used to unNormalize all standardized functions from standardizeCompute.
        This returns the data back to its original state.

        INPUT:
        dataset: The dataset, with labels at the end. NOTE: Ensure labels are at the end
        standardizer: an array of:
            mean: means of every feature in the dataset
            std: the standard deviations of every feature in the dataset

        OUTPUT:
        tounNormalize: the resulting dataset, unstandardized to revert it back to its orginal state.
        '''
        return (tounNormalize*standardizer[1]) + standardizer[0]
