import sys
import os

import pandas as pd

sys.path.append(os.path.dirname(__file__) + '/../')

from matplotlib import pyplot as plt

from utils import *

import numpy as np

def fill_age_train(row, data):
    data["question_id"].append(int(row[0]))
    data["user_id"].append(int(row[1]))
    data["is_correct"].append(int(row[2]))

def get_sub_age(train_data_df, gender_sub):
    merged_age = pd.merge(train_data_df, gender_sub, on='user_id')[['question_id','user_id', 'is_correct']]
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    merged_age.apply(lambda row: fill_age_train(row, data), axis=1)
    return data


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha, g):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    c = data["is_correct"]
    for i in range(len(c)):
        u_id = data["user_id"][i]
        q_id = data["question_id"][i]
        exp_expr = np.exp(alpha[q_id] * (theta[u_id] - beta[q_id]))
        log_lklihood += (1 - c[i]) * np.log(1 - g[q_id]) + \
                        c[i] * np.log(g[u_id] + exp_expr) - np.log(1 + exp_expr)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha, g):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    dp_dtheta = np.zeros(len(theta))
    dp_dbeta = np.zeros(len(beta))
    dp_dalpha = np.zeros(len(alpha))
    c = data["is_correct"]
    for idx in range(len(c)):
        i = data["user_id"][idx]
        j = data["question_id"][idx]
        exp_expr = np.exp(alpha[j] * (theta[i] - beta[j]))
        try:
            dp_dtheta[i] += c[idx] * alpha[j] * (exp_expr / (g[j] + exp_expr)) - \
                        alpha[j] * (exp_expr / (1 + exp_expr))
        except:
            pass
        dp_dbeta[j] += - c[idx] * alpha[j] * (exp_expr / (g[j] + exp_expr)) + \
                       alpha[j] * (exp_expr / (1 + exp_expr))
        dp_dalpha[j] += c[idx] * (theta[i] - beta[j]) * (exp_expr / (g[j] + exp_expr)) - \
                        (theta[i] - beta[j]) * (exp_expr / (1 + exp_expr))
    theta = theta + lr * dp_dtheta
    beta = beta + lr * dp_dbeta
    alpha = alpha + lr * dp_dalpha
    # print("theta", theta)
    # print("beta", beta)
    # print("alpha", alpha)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def irt(data, val_data, lr, iterations, g_val):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(542)
    beta = np.zeros(1774)
    alpha = np.ones(1774)
    g = np.zeros(1774)
    g.fill(g_val)

    val_acc_lst = []
    tl_lst = []
    vl_lst = []
    i_lst = []

    theta_lst = []
    beta_lst = []
    alpha_lst = []

    for i in range(iterations):
        i_lst.append(i)
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha, g=g)
        tl_lst.append(neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha, g=g)
        vl_lst.append(val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, g=g)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha, g)
        theta_lst.append(theta)
        beta_lst.append(beta)
        alpha_lst.append(alpha)

    # TODO: You may change the return values to achieve what you want.
    max_iteration = np.argmax(val_acc_lst) + 1

    return theta_lst[max_iteration - 1], beta_lst[max_iteration - 1], \
           alpha_lst[max_iteration - 1], val_acc_lst, max_iteration, \
           tl_lst, vl_lst, i_lst

def get_corrsepond_theta(inital_thetas, row, theta_lst):
    '''

    :param inital_thetas: follow the (theta>=16, theta < 16, theta_nan_age

    '''
    age = row['age']
    if pd.isnull(age):
        theta_lst.append(inital_thetas[2])
    elif age >= 15:
        theta_lst.append(inital_thetas[0])
    else:
        theta_lst.append(inital_thetas[1])

def irt_combined(data, val_data, lr, iterations, g_val, inital_thetas, student_meta):
    theta = []
    student_meta.apply(lambda row: get_corrsepond_theta(inital_thetas, row, theta), axis=1)
    theta = np.array(theta)
    beta = np.zeros(1774)
    alpha = np.ones(1774)
    g = np.zeros(1774)
    g.fill(g_val)

    val_acc_lst = []
    tl_lst = []
    vl_lst = []
    i_lst = []

    theta_lst = []
    beta_lst = []
    alpha_lst = []

    for i in range(iterations):
        i_lst.append(i)
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha, g=g)
        tl_lst.append(neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha, g=g)
        vl_lst.append(val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, g=g)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha, g)
        theta_lst.append(theta)
        beta_lst.append(beta)
        alpha_lst.append(alpha)

    max_iteration = np.argmax(val_acc_lst) + 1

    return theta_lst[max_iteration - 1], beta_lst[max_iteration - 1], \
           alpha_lst[max_iteration - 1], val_acc_lst, max_iteration, \
           tl_lst, vl_lst, i_lst

def evaluate(data, theta, beta, alpha, g):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q]))
        p_a = g[q] + (1 - g[q]) / (1 + np.exp(-alpha[q] * (theta[u] - beta[q])))
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    train_data_df = pd.read_csv("../data/train_data.csv")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    student_meta = pd.read_csv('../data/student_ages.csv')
    # # Set hyperparameters to tune
    # learning_rates = [0.01, 0.011, 0.013, 0.014, 0.015]
    # iteration = 30
    # g_val = 0.01
    # max_dict = {}

    # # Train using irt
    # for learning_rate in learning_rates:
    #     theta, beta, alpha, val_acc_lst, max_iteration, tl_lst, vl_lst, i_lst = \
    #         irt(train_data, train_data, learning_rate, iteration, g_val)
    #     max_dict[learning_rate] = (tl_lst, vl_lst, i_lst, theta, beta, alpha,
    #                                val_acc_lst[max_iteration - 1],
    #                                max_iteration)
    #     print("Learning rate: ", learning_rate)
    #     print("Accuracy: ", val_acc_lst)
    #     print("Max accuracy iteration: ", max_iteration)
    #     print("\n")
    #
    # # Choose the hyperparameters with maximum accuracy
    # max_lr = 0
    # max_val = 0.
    # max_itr = 0
    # for lr in max_dict.keys():
    #     if max_dict[lr][6] >= max_val:
    #         max_lr = lr
    #         max_val = max_dict[lr][6]
    #         max_itr = max_dict[lr][7]
    # print("Chosen learning rate: ", max_lr)
    # print("Chosen iteration: ", max_itr)
    # chosen_tl_lst = max_dict[max_lr][0]
    # chosen_vl_lst = max_dict[max_lr][1]
    # chosen_i_lst = max_dict[max_lr][2]
    # chosen_theta = max_dict[max_lr][3]
    # chosen_beta = max_dict[max_lr][4]
    # chosen_alpha = max_dict[max_lr][5]

    # lr = 0.014
    # it = 37
    #
    # print("Learning rate: ", lr)
    # print("Iteration: ", it)

    lr = 0.014
    it = 29
    g_val = 0.01
    # get the the theta for student age >= 16
    train_greater_16 = get_sub_age(train_data_df, student_meta[student_meta['age'] >= 17])
    theta_fv_greater_16, beta_fv, alpha_fv, val_acc_lst_fv, max_iteration_fv, tl_lst_fv, \
    vl_lst_fv, i_lst_fv = irt(train_greater_16, val_data, lr, it, g_val)
    # get the the theta for student age < 16
    train_less_16 = get_sub_age(train_data_df, student_meta[student_meta['age'] < 17])
    theta_fv_less_16, beta_fv, alpha_fv, val_acc_lst_fv, max_iteration_fv, tl_lst_fv, \
    vl_lst_fv, i_lst_fv = irt(train_less_16, val_data, lr, it, g_val)
    # get the average theta for >=16
    theta_fv_greater_16 = sum(theta_fv_greater_16)/len(theta_fv_greater_16)
    # get the average theta for < 16
    theta_fv_less_16 = sum(theta_fv_less_16)/len(theta_fv_less_16)

    # Final train accuracy
    theta_fv, beta_fv, alpha_fv, val_acc_lst_fv, max_iteration_fv, tl_lst_fv, \
    vl_lst_fv, i_lst_fv = irt_combined(train_data, train_data, lr, it, g_val, [theta_fv_greater_16, theta_fv_less_16, 0],
                                       student_meta)
    print("Final train accuracy: ", val_acc_lst_fv[-1])

    # Final validation accuracy
    theta_fv, beta_fv, alpha_fv, val_acc_lst_fv, max_iteration_fv, tl_lst_fv, \
    vl_lst_fv, i_lst_fv = irt_combined(train_data, val_data, lr, it, g_val, [theta_fv_greater_16, theta_fv_less_16, 0], student_meta)
    print("Final validation accuracy: ", val_acc_lst_fv[-1])

    # Final test accuracy
    # theta_ft, beta_ft, alpha_ft, val_acc_lst_ft, max_iteration_ft, tl_lst_ft, \
    # vl_lst_ft, i_lst_ft = irt(train_data, test_data, lr, it, g_val)
    # print("Final test accuracy: ", val_acc_lst_ft[-1])
    theta_ft, beta_ft, alpha_ft, val_acc_lst_ft, max_iteration_ft, tl_lst_ft, \
    vl_lst_fv, i_lst_fv = irt_combined(train_data, test_data, lr, it, g_val, [theta_fv_greater_16, theta_fv_less_16, 0], student_meta)
    print("Final test accuracy: ", val_acc_lst_ft[-1])


if __name__ == "__main__":
    main()
