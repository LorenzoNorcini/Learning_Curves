from Cross_Validation import Cross_Validation as crv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import time


class Learning_Curve:

    """
    This class is contains utility methods to compute and plot learning curves
    """

    """ Takes the following parameters as an input:
        learner: a SKLearn Classifier (SKLearn Classifier)
        k: number of iteration of Cross-Validation (float)
        examples: the Bag of Words (sparse matrix of integers)
        labels: labels for each sample (list of integers)
        sizes: sizes to be tested (list of floats between 0 and 1)
        Returns:
        trains_sizes: the tested sizes (list of integers)
        train_scores: the scores achieved on the train set (matrix of floats between 0 and 100)
        test_scores: the scores achieved on the test set (matrix of floats between 0 and 100)
    """
    @staticmethod
    def Learning_Curve(learner, k, examples, labels, sizes, type="shuffle"):
        train_scores = []
        test_scores = []
        train_sizes = []
        examples, labels = shuffle(examples, labels, random_state=int(time.time()))
        for s in sizes:
            size = s*examples.shape[0]
            data_slice, labels_slice = shuffle(examples[:size], labels[:size], random_state=int(time.time()))
            if type == "shuffle":
                score = crv.Shuffle_Cross_Validation(learner, k, data_slice, labels_slice, 0.1)
            elif type == "k-fold":
                score = crv.K_Fold_Cross_Validation(learner, k, data_slice, labels_slice)
            train_scores.append(score[0])
            test_scores.append(score[1])
            train_sizes.append(size)
        return train_sizes, train_scores, test_scores

    """ Takes the following parameters as an input:
        trains_sizes: the tested sizes (list of integers)
        avg_train_scores: the scores achieved on the train set for each size(list of floats between 0 and 100)
        avg_test_scores: the scores achieved on the test set for each size(list of floats between 0 and 100)
        test_std_deviation: the standard deviation on the test set for each size (list of floats)
        train_std_deviation: the standard deviation on the train set for each size (list of floats)
        name: the choosen name (string)
        Returns:
        Nothing.
        Plots the learning curve with the given data
    """
    @staticmethod
    def plot_curve(train_size, avg_test_scores, avg_train_scores, test_std_deviation, train_std_deviation, name):
        print name
        for i in zip(train_size, avg_test_scores):
            print i
        for j in zip(train_size, avg_train_scores):
            print j
        plt.figure()
        plt.plot(train_size, avg_test_scores, 'o-', label=name + " Test", color='blue',)
        plt.plot(train_size, avg_train_scores, 'o-', label=name + " Train", color='green')
        plt.fill_between(train_size, [x[0]+x[1] for x in zip(avg_test_scores, test_std_deviation)],
                 [x[0]-x[1] for x in zip(avg_test_scores, test_std_deviation)], color='blue', alpha=0.2)
        plt.fill_between(train_size, [x[0]+x[1] for x in zip(avg_train_scores, train_std_deviation)],
                 [x[0]-x[1] for x in zip(avg_train_scores, train_std_deviation)], color='green', alpha=0.2)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=1, mode="expand", borderaxespad=0.)
        plt.axis([0, 20000, 0.25, 1.0])
        plt.xticks(range(0, 20000, 1000))
        plt.yticks(np.arange(0.30, 1.0, 0.1))
        for k in range(0, len(train_size)):
            plt.text(train_size[k], avg_test_scores[k], str(round(avg_test_scores[k], 4)))
            plt.text(train_size[k], avg_train_scores[k], str(round(avg_train_scores[k], 4)))
