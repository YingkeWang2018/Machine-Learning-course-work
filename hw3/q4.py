'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.special import logsumexp


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        digits = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(digits, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        digits = data.get_digits_by_label(train_data, train_labels, i)
        i_count = digits.shape[0]
        # broadcast the meann to fit the digits
        i_mean = np.array([means[i]] * i_count)
        digit_diff = digits - i_mean
        covariances[i] = 1/i_count * np.matmul(digit_diff.transpose(), digit_diff)
        covariances[i] += 0.01 * np.identity(64)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n, d = digits.shape
    # initalize the log likelihood
    log_likelihood = np.zeros((n, 10))
    for i in range(10):
        multi = np.multiply(((digits - means[i]) @ np.linalg.inv(covariances[i])), (digits-means[i]))
        multi = np.sum(multi, axis=1)
        log_likelihood_i = (-d/2) * np.log(2 * np.pi) - 1/2 * np.log((np.linalg.det(covariances[i]))) - 1/2 * multi
        log_likelihood[:, i] = log_likelihood_i
    return log_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    condi_likelihood = np.zeros((digits.shape[0], 10))
    generative = generative_likelihood(digits, means, covariances)
    p_y = 0.1
    for example_i in range(generative.shape[0]):
        example = generative[example_i]
        for target in range(10):
            condi_likelihood[example_i][target] = example[target] + np.log(p_y) - logsumexp(example + np.log(p_y))
    return condi_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    sum_ = 0
    for i in range(digits.shape[0]):
        sum_ += cond_likelihood[i][int(labels[i])]
    return sum_/labels.shape[0]

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def data_accuracy(predictions, labels):
    diff = predictions - labels
    diff_index = np.nonzero(diff)[0]
    return (len(diff) - len(diff_index))/ len(diff)

def plot_eigens(covariances):
    for i in range(10):
        eigenvalues, eigenvectors = np.linalg.eig(covariances[i])
        max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
        max_eigenvector = max_eigenvector.reshape((8, 8))
        plt.subplot(2, 5, i+1)
        plt.imshow(max_eigenvector)
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    avg_log_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print(f'average log-likelihood for training set is {avg_log_train}')
    avg_log_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(f'average log-likelihood for training set is {avg_log_test}')
    prediction_train = classify_data(train_data, means, covariances)
    accuracy_train = data_accuracy(prediction_train, train_labels)
    print(f'training set accuracy is {accuracy_train}')
    prediction_test = classify_data(test_data, means, covariances)
    accuracy_test = data_accuracy(prediction_test, test_labels)
    print(f'test set accuracy is {accuracy_test}')
    plot_eigens(covariances)


if __name__ == '__main__':
    main()
