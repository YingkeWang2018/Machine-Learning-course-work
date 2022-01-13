from q3.l2_distance import l2_distance
from q3.utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    k_lst = [1, 3, 5, 7, 9]
    classification_lst = []
    for k in k_lst:
        valid_prediction = knn(k, train_inputs, train_targets, valid_inputs)
        # calculate the average correct guess
        classification = 1 - np.mean((valid_prediction - valid_targets) ** 2)
        classification_lst.append(classification)
    # plot the the classification
    plt.plot(k_lst, classification_lst)
    plt.xlabel('k')
    plt.ylabel('classification rate')
    plt.title('Classification rate on validation set')
    plt.savefig('validation_rate.png')
    # calculate test classification rate
    k_test_lst = [3, 5, 7]
    for k in k_test_lst:
        test_prediction = knn(k, train_inputs, train_targets, test_inputs)
        classification = 1 - np.mean((test_prediction - test_targets) ** 2)
        print(f'test classification rate for k = {k} is {classification}')


if __name__ == "__main__":
    run_knn()
