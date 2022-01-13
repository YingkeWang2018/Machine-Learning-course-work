from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from student_prepro import prepro_main
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from refact_subject_precent import get_percentile_csv


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()

    df_student = prepro_main()
    df_student = df_student.sort_values(by=["user_id"])
    np_students = df_student[["gender", "age_adj", "premium_pupil_adj"]].to_numpy()
    train_matrix = np.hstack((train_matrix, np_students))

    df_subjects = get_percentile_csv()
    df_subjects = df_subjects.sort_values(by=["user_id"])
    np_subjects = df_subjects.drop('user_id', axis=1).to_numpy()

    train_matrix = np.hstack((train_matrix, np_subjects))

    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, student_vectorsize, k=100):
        """ Initialize a class AutoEncoder.

        :param student_vectorsize: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(student_vectorsize, k)
        self.o = nn.Linear(k, 200)
        self.o2 = nn.Linear(200, 200)
        self.p = nn.Linear(200, k)
        #self.h = nn.Linear(k, num_question-3-388)
        self.h = nn.Linear(k, student_vectorsize)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2

        o_w_norm = torch.norm(self.o.weight, 2) ** 2
        p_w_norm = torch.norm(self.p.weight, 2) ** 2
        o2_w_norm = torch.norm(self.o2.weight, 2) ** 2
        return g_w_norm + h_w_norm + o_w_norm + p_w_norm + o2_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        l1=torch.sigmoid(self.g(inputs))
        l2=torch.sigmoid(self.o(l1))
        l3=torch.sigmoid(self.o2(l2))
        l4=torch.sigmoid(self.p(l3))
        out = torch.sigmoid(self.h(l4))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: steps,train_loss and validation_acc
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.002)   #with L2
    num_student = train_data.shape[0]

    steps = []
    train_losses = []
    validation_acc = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())

            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)

            # loss = torch.sum((output - target) ** 2. )+((lamb/2)*(model.get_weight_norm()))  #3e add L2 regularization
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc, _ = evaluate(model, zero_train_data, valid_data)
        steps.append(epoch)
        train_losses.append(train_loss)
        validation_acc.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    return steps, train_losses, validation_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0
    guesses = []

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
        guesses.append(guess)
    return correct / float(total),guesses

if __name__ == '__main__':
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Set model hyperparameters.
    test_acc = []

    k = 300
    model = AutoEncoder(train_matrix.shape[1], k)

    # Set optimization hyperparameters.
    lr = 0.02
    num_epoch = 70
    lamb = 0.01

    steps, train_losses, validation_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    final_test_accuracy, guesses = evaluate(model, zero_train_matrix, test_data)
    print(final_test_accuracy)