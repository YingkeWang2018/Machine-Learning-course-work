from q3.check_grad import check_grad
from q3.utils import *
from q3.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    print(f"small length {len(train_targets)}")
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    N, M = train_inputs.shape
    weights = np.zeros(M + 1).reshape((M+1, 1))
    hyperparameters = {
        "learning_rate": 0.06,
        "weight_regularization": 0.,
        "num_iterations": 80
    }
    run_check_grad(hyperparameters)
    train_ces = []
    validation_ces = []
    train_accuracies = []
    validation_accuracies = []
    # record the accuracy and ce for 0 iteration
    train_entropy, train_accuracy = evaluate(train_targets, logistic_predict(weights, train_inputs))
    validation_entropy, validation_accuracy = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
    train_ces.append(train_entropy)
    validation_ces.append(validation_entropy)
    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)

    for t in range(hyperparameters["num_iterations"]):
        # get the derivative
        df = logistic(weights, train_inputs, train_targets, hyperparameters)[1]
        weights -= hyperparameters["learning_rate"] * df
        # calculate entropy and accuracy
        train_entropy, train_accuracy = evaluate(train_targets, logistic_predict(weights, train_inputs))
        validation_entropy, validation_accuracy = evaluate(valid_targets, logistic_predict(weights, valid_inputs))
        train_ces.append(train_entropy)
        validation_ces.append(validation_entropy)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
    # save the accuracy plot to choose the optimal num of iterations
    plt.plot(list(range(hyperparameters["num_iterations"] + 1)), train_accuracies, label="train accuracy")
    plt.plot(list(range(hyperparameters["num_iterations"] + 1)), validation_accuracies, label="valid accuracy")

    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    # plot ces
    plt.plot(list(range(hyperparameters["num_iterations"] + 1)), train_ces, label="train ces")
    plt.plot(list(range(hyperparameters["num_iterations"] + 1)), validation_ces, label="valid ces")
    plt.xlabel("iterations")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()
    # from the test set and get the CE and cross entropy
    test_entropy, test_accuracy = evaluate(test_targets, logistic_predict(weights, test_inputs))
    print(f"the train classification error is {1 - train_accuracies[-1]}")
    print(f"the valid classification error is {1 - validation_accuracies[-1]}")
    print(f"the test classification error is {1 - test_accuracy}")
    print(f"the train CE is {train_ces[-1]}")
    print(f"the valid CE is {validation_ces[-1]}")
    print(f"the test CE is {test_entropy}")






def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
