from Cross_Validation import Cross_Validation as cv
from sklearn.utils import shuffle
import numpy as np

class Learning_Curve:

    @staticmethod
    def Learning_Curve(learner, k, examples, labels, sizes):
        train_scores = []
        test_scores = []
        train_sizes = []
        examples, labels = shuffle(examples, labels, random_state=0)
        for s in sizes:
            size = s*examples.shape[0]
            data_slice = examples[:size]
            labels_slice = labels[:size]
            score = cv.Cross_Validation(learner, k, data_slice, labels_slice)
            train_scores.append(score[0])
            test_scores.append(score[1])
            train_sizes.append(size)
        return train_sizes, train_scores, test_scores

