from sklearn.feature_extraction.text import CountVectorizer
from Data_Preprocessing import Data_Preprocessor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
import matplotlib.pyplot as plt
import numpy as np
import time

full_dataset_path = "20news"

dp = Data_Preprocessor()
vectorizer = CountVectorizer(tokenizer=dp.tokenize, stop_words='english', lowercase=True, min_df=2, analyzer="word")
clfMNB = MultinomialNB(alpha=.01)
clfBNB = BernoulliNB(alpha=.01)
clfPer = Perceptron(alpha=.01)
print "Loading data..."
start = time.time()
data, labels = dp.raw_to_vector(full_dataset_path, ["header"])
done = time.time()
elapsed = done - start
print "Elapsed: " + str(elapsed)
start = time.time()
print "Vectorizing..."
data = vectorizer.fit_transform(data)
done = time.time()
elapsed = done - start
print "Elapsed: " + str(elapsed)
print "Number of Samples: " + str(data.shape[0])
print "Number of Features: " + str(data.shape[1])
print "Learning Curve:"
shuffle = cross_validation.ShuffleSplit(data.shape[0], n_iter=10, random_state=0)

MNB_train_size, train_scores, test_scores = \
    learning_curve(clfMNB, data, labels, cv=shuffle, n_jobs=4, train_sizes=np.arange(0.01, 1.00, 0.05))

MNB_avg_test_scores = []
MNB_avg_train_scores = []
for k in range(0, len(test_scores)):
    MNB_avg_test_scores.append(np.average(test_scores[k]))
    MNB_avg_train_scores.append(np.average(train_scores[k]))

BNB_train_size, train_scores, test_scores = \
    learning_curve(clfBNB, data, labels, cv=shuffle, n_jobs=4, train_sizes=np.arange(0.01, 1.00, 0.05))

BNB_avg_test_scores = []
BNB_avg_train_scores = []
for k in range(0, len(test_scores)):
    BNB_avg_test_scores.append(np.average(test_scores[k]))
    BNB_avg_train_scores.append(np.average(train_scores[k]))

Per_train_size, train_scores, test_scores = \
    learning_curve(clfPer, data, labels, cv=shuffle, n_jobs=4, train_sizes=np.arange(0.01, 1.00, 0.05))

Per_avg_test_scores = []
Per_avg_train_scores = []
for k in range(0, len(test_scores)):
    Per_avg_test_scores.append(np.average(test_scores[k]))
    Per_avg_train_scores.append(np.average(train_scores[k]))

plt.plot(MNB_train_size, MNB_avg_test_scores, label="Multinomial Naive Bayes score")
plt.plot(BNB_train_size, BNB_avg_test_scores, label="Bernoulli Naive Bayes score")
plt.plot(Per_train_size, Per_avg_test_scores, label="Perceptron score")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1
           , ncol=1, mode="expand", borderaxespad=0.)
plt.axis([0, 20000, 0.1, 1])
plt.xticks(range(0, 20000, 1000))
plt.yticks(np.arange(0, 1, 0.1))
for k in range(0, len(MNB_train_size), 7):
    plt.text(MNB_train_size[k], MNB_avg_test_scores[k], str(MNB_avg_test_scores[k]))
    plt.text(Per_train_size[k], Per_avg_test_scores[k], str(Per_avg_test_scores[k]))
    plt.text(BNB_train_size[k], BNB_avg_test_scores[k], str(BNB_avg_test_scores[k]))
plt.show()
