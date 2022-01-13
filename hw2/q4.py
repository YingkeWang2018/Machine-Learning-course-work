# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import math
from scipy.special import logsumexp
np.random.seed(0)
from sklearn.model_selection import train_test_split

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist



def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N = x_train.shape[0]
    d = x_train.shape[1]
    test_datum = test_datum.reshape((1, -1))
    dist = l2(test_datum, x_train)
    diagonal_lst = []
    deno = np.sum(np.exp([(-d / (2 * tau ** 2)) for d in dist[0, :]]))
    for i in range(N):
        diagonal_lst.append((np.exp(-dist[0][i] / (2 * tau ** 2))) / deno)
    # get the diagonal matrix
    A = np.zeros((N, N), float)
    np.fill_diagonal(A, diagonal_lst)
    # calculate w*
    a = np.matmul(np.matmul(x_train.T, A), x_train) + np.identity(d) * lam
    b = np.matmul(np.matmul(x_train.T, A), y_train.reshape(-1, 1))
    w = np.linalg.solve(a, b)
    # calculate yhat
    w = w.reshape((14, 1))
    y_hat = np.matmul(test_datum, w)
    return y_hat[0][0]


def calculate_loss(data_train, label_train, datas, targets, tau):
    N = datas.shape[0]
    predicted = []
    for i in range(N):
        test_datum = datas[i]
        y_hat = LRLS(test_datum, data_train, label_train, tau)
        predicted.append(y_hat)
    predicted = np.array(predicted)
    loss = (np.sum((predicted - targets) ** 2)) / N
    return loss


def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    data_train, data_valid, label_train, label_valid = train_test_split(x, y, test_size=val_frac)
    train_losses = []
    test_losses = []
    for tau in taus:
        train_losses.append(calculate_loss(data_train, label_train, data_train, label_train, tau))
        test_losses.append(calculate_loss(data_train, label_train, data_valid, label_valid, tau))
    return train_losses, test_losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(taus, test_losses)
    plt.show()
