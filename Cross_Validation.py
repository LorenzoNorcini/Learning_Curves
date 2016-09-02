from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import numpy as np
import time

class Cross_Validation:

    """
    This class contains utility methods to compute Cross Validation
    """

    """ Takes the following parameters as an input:
        vector: vector to be partitioned in train_set and test_set (either a numpy array or a sparse matrix)
        fold: index of the current portion to use as test set (integer)
        k: size/k gives the size of the test set (integer)
        Returns:
        training: the portion of vector to use as training (either a numpy array or a sparse matrix)
        validation: the portion of vector to use as validation (either a numpy array or a sparse matrix)
    """
    @staticmethod
    def partition(vector, fold, k):
        size = vector.shape[0]
        start = (size/k)*fold
        end = (size/k)*(fold+1)
        validation = vector[start:end]
        if str(type(vector)) == "<class 'scipy.sparse.csr.csr_matrix'>":
            indices = range(start, end)
            mask = np.ones(vector.shape[0], dtype=bool)
            mask[indices] = False
            training = vector[mask]
        elif str(type(vector)) == "<type 'numpy.ndarray'>":
            training = np.concatenate((vector[:start], vector[end:]))
        else:
            return "Error, unexpected data type"
        return training, validation

    """ Takes the following parameters as an input:
        learner: a SKLearn Classifier (SKLearn Classifier)
        k: number of total folds of Cross-Validation (integer)
        examples: the Bag of Words (sparse matrix of integers)
        labels: labels for each sample (list of integers)
        random_split: determines whether to use progressive splits of the set or random split (boolean)
        Returns:
        train_folds_score: the scores on the train set (list of floats)
        validation_folds_score: the scores on the test set (list of floats)
    """
    @staticmethod
    def K_Fold_Cross_Validation(learner, k, examples, labels, random_split=False):
        train_folds_score = []
        validation_folds_score = []
        examples, labels = shuffle(examples, labels, random_state=int(time.time()))
        for fold in range(0, k):
            if random_split:
                training_set, validation_set, training_labels, validation_labels = \
                    train_test_split(examples, labels, test_size=1./k, random_state=int(time.time()+fold))
            else:
                training_set, validation_set = Cross_Validation.partition(examples, fold, k)
                training_labels, validation_labels = Cross_Validation.partition(labels, fold, k)
            learner.fit(training_set, training_labels)
            training_predicted = learner.predict(training_set)
            validation_predicted = learner.predict(validation_set)
            train_folds_score.append(metrics.accuracy_score(training_labels, training_predicted))
            validation_folds_score.append(metrics.accuracy_score(validation_labels, validation_predicted))
        return train_folds_score, validation_folds_score

    """ Takes the following parameters as an input:
        learner: a SKLearn Classifier (SKLearn Classifier)
        iters: number of iteration of Cross-Validation (float)
        examples: the Bag of Words (sparse matrix of integers)
        labels: labels for each sample (list of integers)
        test_size: test set size for each iteration (float between 0 and 1)
        Returns:
        train_iter_score: the scores on the train set (list of floats)
        validation_iter_score: the scores on the test set (list of floats)
    """
    @staticmethod
    def Shuffle_Cross_Validation(learner, iters, examples, labels, test_size=0.1):
        train_iter_score = []
        validation_iter_score = []
        for iter in range(0, iters):
            training_set, validation_set, training_labels, validation_labels = \
                    train_test_split(examples, labels, test_size=test_size, random_state=int(time.time()+iter))
            learner.fit(training_set, training_labels)
            training_predicted = learner.predict(training_set)
            validation_predicted = learner.predict(validation_set)
            train_iter_score.append(metrics.accuracy_score(training_labels, training_predicted))
            validation_iter_score.append(metrics.accuracy_score(validation_labels, validation_predicted))
        return train_iter_score, validation_iter_score

    """ Takes the following parameters as an input:
        vector: score results (list of list of floats)
        Returns:
        the average and the standard deviation (lists of floats)
    """
    @staticmethod
    def average_and_std_deviation(vector):
        return [np.average(ts) for ts in vector], [np.std(ts) for ts in vector]
