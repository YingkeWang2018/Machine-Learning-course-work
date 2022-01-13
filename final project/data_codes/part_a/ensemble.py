import sys
import os

sys.path.append(os.path.dirname(__file__) + '/../')

from utils import *
from knn import *
from item_response import *
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def resample_data(train_data):
    c = train_data["is_correct"]
    new_user = []
    new_question = []
    new_correct = []
    for i in range(len(c)):
        rand_idx = np.random.randint(len(c))
        new_user.append(train_data["user_id"][rand_idx])
        new_question.append(train_data["question_id"][rand_idx])
        new_correct.append(c[rand_idx])
    result = {"user_id": new_user, "question_id": new_question,
              "is_correct": new_correct}
    return result


def resample_matrix(data_dict):
    new_matrix = np.zeros((542, 1774))
    new_matrix.fill(np.nan)
    c = data_dict["is_correct"]
    for idx in range(len(c)):
        student_id = data_dict["user_id"][idx]
        question_id = data_dict["question_id"][idx]
        correctness = data_dict["is_correct"][idx]
        new_matrix[student_id, question_id] = correctness
    return new_matrix


def main():
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data")
    # print("sparse matrix", sparse_matrix)
    # print("shape", sparse_matrix.shape)
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    train1 = resample_data(train_data)
    train2 = resample_data(train_data)
    train3 = resample_data(train_data)

    new_m1 = resample_matrix(train1)
    new_m2 = resample_matrix(train2)
    new_m3 = resample_matrix(train3)
    # print("new matrix", new_m1)

    val_acc = []
    test_acc = []
    val_predictions = []
    test_predictions = []

    # # Three KNN models
    # print("Bagging with three KNN models")
    #
    # print("\nKNN user based model with k = 11")
    # nbrs1 = KNNImputer(n_neighbors=11)
    # mat1 = nbrs1.fit_transform(new_m1)
    # val_acc1 = sparse_matrix_evaluate(val_data, mat1)
    # test_acc1 = sparse_matrix_evaluate(test_data, mat1)
    # # val_acc1 = knn_impute_by_user(new_m1, val_data, 11)
    # # test_acc1 = knn_impute_by_user(new_m1, test_data, 11)
    # print("Validation Accuracy: ", val_acc1)
    # print("Test Accuracy: ", test_acc1)
    # val_acc.append(val_acc1)
    # test_acc.append(test_acc1)
    #
    # print("\nKNN item based model with k = 21")
    # nbrs2 = KNNImputer(n_neighbors=21)
    # mat2 = nbrs2.fit_transform(new_m2.transpose())
    # val_acc2 = sparse_matrix_evaluate(val_data, mat2.transpose())
    # test_acc2 = sparse_matrix_evaluate(test_data, mat2.transpose())
    # # val_acc2 = knn_impute_by_item(new_m2, val_data, 21)
    # # test_acc2 = knn_impute_by_item(new_m2, test_data, 21)
    # print("Validation Accuracy: ", val_acc2)
    # print("Test Accuracy: ", test_acc2)
    # val_acc.append(val_acc2)
    # test_acc.append(test_acc2)
    #
    # print("\nKNN item based model with k = 26")
    # nbrs3 = KNNImputer(n_neighbors=26)
    # mat3 = nbrs3.fit_transform(new_m3.transpose())
    # val_acc3 = sparse_matrix_evaluate(val_data, mat3.transpose())
    # test_acc3 = sparse_matrix_evaluate(test_data, mat3.transpose())
    # # val_acc3 = knn_impute_by_item(new_m3, val_data, 26)
    # # test_acc3 = knn_impute_by_item(new_m3, test_data, 26)
    # print("Validation Accuracy: ", val_acc3)
    # print("Test Accuracy: ", test_acc3)
    # val_acc.append(val_acc3)
    # test_acc.append(test_acc3)
    #
    # total_prediction_val = 0
    # total_accurate_val = 0
    # for i in range(len(val_data["is_correct"])):
    #     cur_user_id = val_data["user_id"][i]
    #     cur_question_id = val_data["question_id"][i]
    #     val_pred = (mat1[cur_user_id, cur_question_id] + mat2.transpose()[cur_user_id, cur_question_id] + mat3.transpose()[cur_user_id, cur_question_id]) / 3
    #     if val_pred >= 0.5 and val_data["is_correct"][i]:
    #         total_accurate_val += 1
    #     if val_pred < 0.5 and not val_data["is_correct"][i]:
    #         total_accurate_val += 1
    #     total_prediction_val += 1
    # final_val_acc = total_accurate_val / float(total_prediction_val)
    #
    # total_prediction_test = 0
    # total_accurate_test = 0
    # for j in range(len(test_data["is_correct"])):
    #     cur_user_id = test_data["user_id"][j]
    #     cur_question_id = test_data["question_id"][j]
    #     test_pred = (mat1[cur_user_id, cur_question_id] + mat2.transpose()[cur_user_id, cur_question_id] + mat3.transpose()[cur_user_id, cur_question_id]) / 3
    #     if test_pred >= 0.5 and test_data["is_correct"][j]:
    #         total_accurate_test += 1
    #     if test_pred < 0.5 and not test_data["is_correct"][j]:
    #         total_accurate_test += 1
    #     total_prediction_test += 1
    # final_test_acc = total_accurate_test / float(total_prediction_test)

    # # Three IRT models
    # print("Bagging with three IRT models")
    # lr_irt = [0.01, 0.015, 0.011]
    # iter_irt = [8, 10, 13]
    # theta_val_lst = []
    # beta_val_lst = []
    # theta_test_lst = []
    # beta_test_lst = []
    # train_data_irt = [train1, train2, train3]
    # for i in range(len(lr_irt)):
    #     print("\nIRT model with learning rate ", lr_irt[i], " and iteration ", iter_irt[i])
    #     theta_val, beta_val, val_acc_lst_val, max_iteration_val, tl_lst_val, \
    #     vl_lst_val, i_lst_val = irt(train_data_irt[i], val_data,
    #                                 lr_irt[i], iter_irt[i])
    #     theta_test, beta_test, val_acc_lst_test, max_iteration_test, tl_lst_test, \
    #     vl_lst_test, i_lst_test = irt(train_data_irt[i], test_data,
    #                                   lr_irt[i], iter_irt[i])
    #     print("Validation Accuracy: ", val_acc_lst_val[max_iteration_val - 1])
    #     print("Test Accuracy: ", val_acc_lst_test[max_iteration_test - 1])
    #     val_acc.append(val_acc_lst_val[max_iteration_val - 1])
    #     test_acc.append(val_acc_lst_test[max_iteration_test - 1])
    #     theta_val_lst.append(theta_val)
    #     beta_val_lst.append(beta_val)
    #     theta_test_lst.append(theta_test)
    #     beta_test_lst.append(beta_test)
    #
    # for j in range(3):
    #     val_pred = []
    #     test_pred = []
    #     for i1, q1 in enumerate(val_data["question_id"]):
    #         u1 = val_data["user_id"][i1]
    #         x1 = (theta_val_lst[j][u1] - beta_val_lst[j][q1]).sum()
    #         p_a1 = sigmoid(x1)
    #         val_pred.append(p_a1)
    #     for i2, q2 in enumerate(test_data["question_id"]):
    #         u2 = test_data["user_id"][i2]
    #         x2 = (theta_val_lst[j][u2] - beta_val_lst[j][q2]).sum()
    #         p_a2 = sigmoid(x2)
    #         test_pred.append(p_a2)
    #     val_predictions.append(val_pred)
    #     test_predictions.append(test_pred)
    #
    # val_prediction = []
    # test_prediction = []
    # for k1 in range(len(val_data["user_id"])):
    #     val_prediction.append((val_predictions[0][k1] + val_predictions[1][k1] + val_predictions[2][k1]) / 3 >= 0.5)
    # final_val_acc = np.sum((val_data["is_correct"] == np.array(val_prediction))) \
    #                 / len(val_data["is_correct"])
    # for k2 in range(len(test_data["user_id"])):
    #     test_prediction.append((test_predictions[0][k2] + test_predictions[1][k2] + test_predictions[2][k2]) / 3 >= 0.5)
    # final_test_acc = np.sum((test_data["is_correct"] == np.array(test_prediction))) \
    #                  / len(test_data["is_correct"])

    #

    # Two KNN models and one IRT model
    print("Bagging with two KNN models and one IRT model")

    print("\nKNN user based model with k = 11")
    nbrs1 = KNNImputer(n_neighbors=11)
    mat1 = nbrs1.fit_transform(new_m1)
    val_acc1 = sparse_matrix_evaluate(val_data, mat1)
    test_acc1 = sparse_matrix_evaluate(test_data, mat1)
    # val_acc1 = knn_impute_by_user(new_m1, val_data, 11)
    # test_acc1 = knn_impute_by_user(new_m1, test_data, 11)
    print("Validation Accuracy: ", val_acc1)
    print("Test Accuracy: ", test_acc1)
    val_acc.append(val_acc1)
    test_acc.append(test_acc1)

    print("\nKNN item based model with k = 21")
    nbrs2 = KNNImputer(n_neighbors=21)
    mat2 = nbrs2.fit_transform(new_m2.transpose())
    val_acc2 = sparse_matrix_evaluate(val_data, mat2.transpose())
    test_acc2 = sparse_matrix_evaluate(test_data, mat2.transpose())
    # val_acc2 = knn_impute_by_item(new_m2, val_data, 21)
    # test_acc2 = knn_impute_by_item(new_m2, test_data, 21)
    print("Validation Accuracy: ", val_acc2)
    print("Test Accuracy: ", test_acc2)
    val_acc.append(val_acc2)
    test_acc.append(test_acc2)

    print("\nIRT model with learning rate 0.015 and iteration 10")
    theta_val, beta_val, val_acc_lst_val, max_iteration_val, tl_lst_val, \
    vl_lst_val, i_lst_val = \
        irt(train1, val_data, 0.015, 10)
    theta_test, beta_test, val_acc_lst_test, max_iteration_test, \
    tl_lst_test, vl_lst_test, i_lst_test = \
        irt(train1, test_data, 0.015, 10)
    print("Validation Accuracy: ", val_acc_lst_val[max_iteration_val - 1])
    print("Test Accuracy: ", val_acc_lst_test[max_iteration_test - 1])
    val_acc.append(val_acc_lst_val[max_iteration_val - 1])
    test_acc.append(val_acc_lst_test[max_iteration_test - 1])

    total_prediction_val = 0
    total_accurate_val = 0
    for i in range(len(val_data["is_correct"])):
        cur_user_id = val_data["user_id"][i]
        cur_question_id = val_data["question_id"][i]
        x = (theta_val[cur_user_id] - beta_val[cur_question_id]).sum()
        p_a = sigmoid(x)
        val_pred = (mat1[cur_user_id, cur_question_id] + mat2.transpose()[cur_user_id, cur_question_id] + p_a) / 3
        if val_pred >= 0.5 and val_data["is_correct"][i]:
            total_accurate_val += 1
        if val_pred < 0.5 and not val_data["is_correct"][i]:
            total_accurate_val += 1
        total_prediction_val += 1
    final_val_acc = total_accurate_val / float(total_prediction_val)

    total_prediction_test = 0
    total_accurate_test = 0
    for j in range(len(test_data["is_correct"])):
        cur_user_id = test_data["user_id"][j]
        cur_question_id = test_data["question_id"][j]
        x = (theta_test[cur_user_id] - beta_test[cur_question_id]).sum()
        p_a = sigmoid(x)
        test_pred = (mat1[cur_user_id, cur_question_id] + mat2.transpose()[cur_user_id, cur_question_id] + p_a) / 3
        if test_pred >= 0.5 and test_data["is_correct"][j]:
            total_accurate_test += 1
        if test_pred < 0.5 and not test_data["is_correct"][j]:
            total_accurate_test += 1
        total_prediction_test += 1
    final_test_acc = total_accurate_test / float(total_prediction_test)

    print("\n")
    print("Final validation accuracy:", final_val_acc)
    print("Final test accuracy:", final_test_acc)


if __name__ == "__main__":
    main()
