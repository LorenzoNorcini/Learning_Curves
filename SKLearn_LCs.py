from sklearn.feature_extraction.text import CountVectorizer
from Data_Preprocessing import Data_Preprocessor as dp
from Cross_Validation import Cross_Validation as crv
from sklearn.learning_curve import learning_curve
from Learning_Curve import Learning_Curve as lc
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import time

first_start = time.time()

full_dataset_path = "20news"

train_sizes = np.arange(0.1, 1.1, 0.1)
random_samples = 100

jobs = 4

revectorize = True

# Initializes the Vectorizer that will turn the raw data into the Bag of Words model
vectorizer = CountVectorizer(stop_words='english', lowercase=True, min_df=2, analyzer="word")

# Initializes the classifiers
clfMNB = MultinomialNB(alpha=.15)
clfBNB = BernoulliNB(alpha=.15)
clfPer = Perceptron(alpha=.15)

if revectorize:
    dp.clean_saved_vector("saved")

try:
    # Loads the Bag of Words from file (if it has already been calculated and saved)
    # NB if changes are made to the preprocessing it's necessary to delete old saves file
    data = joblib.load("saved/vectorized_data.plk")
    labels = joblib.load("saved/labels.plk")
    print "Data vector found... loaded..."
except:
    # If the Bag of Words is not present it's calculated from the the raw DataSet
    print "Loading data..."
    start = time.time()
    # Create vectors of data (still in text form) and labels
    data, labels = dp.raw_to_vector(full_dataset_path, ["header"])
    done = time.time()
    elapsed = done - start
    print "Elapsed: " + str(elapsed)
    start = time.time()
    print "Data vector not found... vectorizing..."
    data = vectorizer.fit_transform(data)                   # Create Bag of Words from data
    joblib.dump(data, "saved/vectorized_data.plk")          # Saves Bag of Words
    joblib.dump(labels, "saved/labels.plk")                 # And Labels
    done = time.time()
    elapsed = done - start
    print "Elapsed: " + str(elapsed)

print "Number of Samples: " + str(data.shape[0])
print "Number of Features: " + str(data.shape[1])

print "Learning Curve:"
start = time.time()
# Initializes the Cross Validator
shuffle = cross_validation.ShuffleSplit(data.shape[0], n_iter=random_samples, random_state=int(time.time()))

# Gets Learning Curve data for MultinomiaNB
MNB_train_size, train_scores, test_scores = \
    learning_curve(clfMNB, data, labels, cv=shuffle, n_jobs=jobs, train_sizes=train_sizes, verbose=1)

# Computes average and standard deviation for the test score and the train score
MNB_avg_test_scores, MNB_test_std_deviation = crv.average_and_std_deviation(test_scores)
MNB_avg_train_scores, MNB_train_std_deviation = crv.average_and_std_deviation(train_scores)

print "Multinomial Naive Bayes has finished"

# Gets Learning Curve data for Bernoulli
BNB_train_size, train_scores, test_scores = \
    learning_curve(clfBNB, data, labels, cv=shuffle, n_jobs=jobs, train_sizes=train_sizes, verbose=1)

# Computes average and standard deviation for the test score and the train score
BNB_avg_test_scores, BNB_test_std_deviation = crv.average_and_std_deviation(test_scores)
BNB_avg_train_scores, BNB_train_std_deviation = crv.average_and_std_deviation(train_scores)

print "Bernoulli Naive Bayes has finished"

# Gets Learning Curve data for Perceptron
Per_train_size, train_scores, test_scores = \
    learning_curve(clfPer, data, labels, cv=shuffle, n_jobs=jobs, train_sizes=train_sizes, verbose=1)

# Computes average and standard deviation for the test score and the train score
Per_avg_test_scores, Per_test_std_deviation = crv.average_and_std_deviation(test_scores)
Per_avg_train_scores, Per_train_std_deviation = crv.average_and_std_deviation(train_scores)

print "Perceptron has finished"


done = time.time()
elapsed = done - start
print "Elapsed: " + str(elapsed)

last_done = time.time()
total_elapsed = done - start
print "Total Elapsed: " + str(total_elapsed)

# Plots the learning curves for both the train set and the test set
# for each of the classifiers

lc.plot_curve(MNB_train_size, MNB_avg_test_scores, MNB_avg_train_scores,
              MNB_test_std_deviation, MNB_train_std_deviation, "Multinomial Naive Bayes")

lc.plot_curve(BNB_train_size, BNB_avg_test_scores, BNB_avg_train_scores,
              BNB_test_std_deviation, BNB_train_std_deviation, "Bernoulli Naive Bayes")

lc.plot_curve(Per_train_size, Per_avg_test_scores, Per_avg_train_scores,
              Per_test_std_deviation, Per_train_std_deviation, "Perceptron")

plt.show()
