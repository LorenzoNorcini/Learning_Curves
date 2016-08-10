from sklearn.feature_extraction.text import CountVectorizer
from Data_Preprocessing import Data_Preprocessor
from sklearn.naive_bayes import MultinomialNB
from Learning_Curve import Learning_Curve as lc
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import time

full_dataset_path = "20news"

train_sizes = np.arange(0.01, 1.00, 0.01)

dp = Data_Preprocessor()
vectorizer = CountVectorizer(tokenizer=dp.tokenize, stop_words='english', lowercase=True, min_df=2, analyzer="word")
clfMNB = MultinomialNB(alpha=.0001)
clfBNB = BernoulliNB(alpha=.0001)
clfPer = Perceptron(alpha=.0001)

try:
    data = joblib.load("saved/vectorized_data.plk")
    labels = joblib.load("saved/labels.plk")
    print "Data vector found... loaded..."
except:
    print "Loading data..."
    start = time.time()
    data, labels = dp.raw_to_vector(full_dataset_path, ["header"])
    done = time.time()
    elapsed = done - start
    print "Elapsed: " + str(elapsed)
    start = time.time()
    print "Data vector not found... vectorizing..."
    data = vectorizer.fit_transform(data)
    joblib.dump(data, "saved/vectorized_data.plk")
    joblib.dump(labels, "saved/labels.plk")
    done = time.time()
    elapsed = done - start
    print "Elapsed: " + str(elapsed)

print "Number of Samples: " + str(data.shape[0])
print "Number of Features: " + str(data.shape[1])

print "Learning Curve:"
start = time.time()

MNB_train_size, train_scores, test_scores = \
    lc.Learning_Curve(clfMNB, 10, data, labels, train_sizes)

MNB_avg_test_scores = [np.average(ts) for ts in test_scores]
MNB_avg_train_scores = [np.average(ts) for ts in train_scores]
MNB_std_deviation = [np.std(ts) for ts in test_scores]

print "Multinomial Naive Bayes has finished"

BNB_train_size, train_scores, test_scores = \
    lc.Learning_Curve(clfBNB, 10, data, labels, train_sizes)

BNB_avg_test_scores = [np.average(ts) for ts in test_scores]
BNB_avg_train_scores = [np.average(ts) for ts in train_scores]
BNB_std_deviation = [np.std(ts) for ts in test_scores]

print "Bernoulli Naive Bayes has finished"

Per_train_size, train_scores, test_scores = \
    lc.Learning_Curve(clfPer, 10, data, labels, train_sizes)

Per_avg_test_scores = [np.average(ts) for ts in test_scores]
Per_avg_train_scores = [np.average(ts) for ts in train_scores]
Per_std_deviation = [np.std(ts) for ts in test_scores]

print "Perceptron has finished"

done = time.time()
elapsed = done - start
print "Elapsed: " + str(elapsed)

plt.plot(MNB_train_size, MNB_avg_test_scores, label="Multinomial Naive Bayes")
plt.plot(BNB_train_size, BNB_avg_test_scores, label="Bernoulli Naive Bayes")
plt.plot(Per_train_size, Per_avg_test_scores, label="Perceptron")

plt.fill_between(MNB_train_size, [x[0]+x[1] for x in zip(MNB_avg_test_scores, MNB_std_deviation)],
                 [x[0]-x[1] for x in zip(MNB_avg_test_scores, MNB_std_deviation)], color='blue', alpha=0.2)
plt.fill_between(BNB_train_size, [x[0]+x[1] for x in zip(BNB_avg_test_scores, BNB_std_deviation)],
                 [x[0]-x[1] for x in zip(BNB_avg_test_scores, BNB_std_deviation)], color='green', alpha=0.2)
plt.fill_between(Per_train_size, [x[0]+x[1] for x in zip(Per_avg_test_scores, Per_std_deviation)],
                 [x[0]-x[1] for x in zip(Per_avg_test_scores, Per_std_deviation)], color='red', alpha=0.2)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=1, mode="expand", borderaxespad=0.)

plt.axis([0, 20000, 0.25, 0.9])
plt.xticks(range(0, 20000, 1000))
plt.yticks(np.arange(0.25, 0.9, 0.1))

for k in range(0, len(MNB_train_size), 5):
    plt.text(MNB_train_size[k], MNB_avg_test_scores[k], str(round(MNB_avg_test_scores[k], 4)))
    plt.text(Per_train_size[k], Per_avg_test_scores[k], str(round(Per_avg_test_scores[k], 4)))
    plt.text(BNB_train_size[k], BNB_avg_test_scores[k], str(round(BNB_avg_test_scores[k], 4)))
plt.show()
