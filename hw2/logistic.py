from q3.utils import sigmoid
import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    N, M = data.shape
    # initialize the data matrix with bias term
    data_added = np.ones((N, M + 1))
    data_added[:, :-1] = data
    # get the prediction
    z = np.matmul(data_added, weights)
    y = sigmoid(z)
    return y

def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    # calculate the cross entropies
    ces = -np.multiply(targets, np.log(y)) - np.multiply(1-targets, np.log(1-y))
    ce = np.mean(ces)
    # get the boolean for greater equal 0.5 (label 1)
    y = y >= 0.5
    y = y.astype(int)
    # calculate the correct rate
    frac_correct = 1 - np.mean((y - targets) ** 2)
    return ce, frac_correct
def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    # get f from the evaluate function
    f, frac_correct = evaluate(targets, y)
    # load the data into a frame
    N, M = data.shape
    # initialize the data matrix with bias term
    data_added = np.ones((N, M + 1))
    data_added[:, :-1] = data
    # calculate the gradient using slide conclusion
    df = 1/N * np.matmul(data_added.transpose(), y - targets)

    return f, df, y
