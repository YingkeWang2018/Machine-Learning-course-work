import sys
import os

sys.path.append(os.path.dirname(__file__) + '/../')

from matplotlib import pyplot as plt

from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
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
        log_lklihood += c[i] * theta[data["user_id"][i]] - \
                        c[i] * beta[data["question_id"][i]] - \
                        np.log(1 + np.exp(theta[data["user_id"][i]] -
                               beta[data["question_id"][i]]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
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
    c = data["is_correct"]
    for idx in range(len(c)):
        i = data["user_id"][idx]
        j = data["question_id"][idx]
        dp_dtheta[i] += c[idx] - sigmoid(theta[i] - beta[j])
        dp_dbeta[j] += - c[idx] + sigmoid(theta[i] - beta[j])
    theta = theta + lr * dp_dtheta
    beta = beta + lr * dp_dbeta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
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
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    tl_lst = []
    vl_lst = []
    i_lst = []

    theta_lst = []
    beta_lst = []

    for i in range(iterations):
        i_lst.append(i)
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        tl_lst.append(neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        vl_lst.append(val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        theta_lst.append(theta)
        beta_lst.append(beta)

    # TODO: You may change the return values to achieve what you want.
    max_iteration = np.argmax(val_acc_lst) + 1

    return theta_lst[max_iteration - 1], beta_lst[max_iteration - 1], val_acc_lst, max_iteration, tl_lst, vl_lst, i_lst


def evaluate(data, theta, beta):
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
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # Set hyperparameters to tune
    learning_rates = [0.015]
    iteration = 12
    max_dict = {}

    # Train using irt
    for learning_rate in learning_rates:
        theta, beta, val_acc_lst, max_iteration, tl_lst, vl_lst, i_lst = \
            irt(train_data, val_data, learning_rate, iteration)
        max_dict[learning_rate] = (tl_lst, vl_lst, i_lst, theta, beta,
                                   val_acc_lst[max_iteration - 1],
                                   max_iteration)
        print("Learning rate: ", learning_rate)
        print("Accuracy: ", val_acc_lst)
        print("Max accuracy iteration: ", max_iteration)
        print("\n")

    # Choose the hyperparameters with maximum accuracy
    max_lr = 0
    max_val = 0.
    max_itr = 0
    for lr in max_dict.keys():
        if max_dict[lr][5] >= max_val:
            max_lr = lr
            max_val = max_dict[lr][5]
            max_itr = max_dict[lr][6]
    print("Chosen learning rate: ", max_lr)
    print("Chosen iteration: ", max_itr)
    chosen_tl_lst = max_dict[max_lr][0]
    chosen_vl_lst = max_dict[max_lr][1]
    chosen_i_lst = max_dict[max_lr][2]
    chosen_theta = max_dict[max_lr][3]
    chosen_beta = max_dict[max_lr][4]

    # Plot the training curve
    # plt.title("Training and Validation Negative log-likelihoods curves")
    # plt.plot([a for a in range(max_dict[max_lr][6])], chosen_tl_lst[:max_dict[max_lr][6]], label="Training set log-likelihood")
    # plt.plot([a for a in range(max_dict[max_lr][6])], chosen_vl_lst[:max_dict[max_lr][6]], label="Validation set log-likelihood")
    # plt.xlabel("# of iterations")
    # plt.ylabel("Negative log-likelihood")
    # plt.legend(loc="upper right")
    # # plt.show()
    # plt.savefig("part_a_2b.png")

    # Final validation accuracy
    theta_fv, beta_fv, val_acc_lst_fv, max_iteration_fv, tl_lst_fv, vl_lst_fv, \
        i_lst_fv = irt(train_data, val_data, max_lr, max_dict[max_lr][6])
    print("Final validation accuracy: ", val_acc_lst_fv[-1])

    # Final test accuracy
    theta_ft, beta_ft, val_acc_lst_ft, max_iteration_ft, tl_lst_ft, vl_lst_ft, \
        i_lst_ft = irt(train_data, test_data, max_lr, max_dict[max_lr][6])
    print("Final test accuracy: ", val_acc_lst_ft[-1])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    plot_theta = np.sort(chosen_theta)
    print(plot_theta.shape)
    for i in range(3):
        q_id = train_data["question_id"][i]
        prob = sigmoid(plot_theta - chosen_beta[q_id])
        label = "Question ID: " + str(q_id)
        plt.plot(plot_theta, prob, label=label)
    plt.xlabel("Theta")
    plt.ylabel("Probability of Correct Response")
    plt.title("Relationship Between Probability of Correct Response and Theta")
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig("part_a_2d.png")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
