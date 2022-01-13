import pandas as pd
import numpy as np
from sklearn import tree
from student_prepro import prepro_main

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from refact_subject_precent import get_question_sub_csv
if __name__ == '__main__':

    df_student = prepro_main()
    df_student = df_student.drop(columns=['age'])
    df_train = pd.read_csv("../data/train_data.csv")
    print(len(df_train))
    print(len(df_train["question_id"].drop_duplicates()))
    print(len(df_train["user_id"].drop_duplicates()))
    df_validate = pd.read_csv("../data/valid_data.csv")
    print(len(df_validate))
    print(len(df_validate["question_id"].drop_duplicates()))
    print(len(df_validate["user_id"].drop_duplicates()))
    df_test = pd.read_csv("../data/test_data.csv")
    df_train = pd.concat([df_train, df_validate])
    df_question = get_question_sub_csv()

    df_train = df_train.merge(df_student, how='inner', on='user_id').merge(df_question, how='inner', on='question_id')

    Xtrain = df_train.drop(columns=['question_id', 'user_id', 'is_correct'])
    Ytrain = list(df_train["is_correct"])
    df_test = df_test.merge(df_student, how='inner', on='user_id').merge(df_question, how='inner', on='question_id')
    Xtest = df_test.drop(columns=['question_id', 'user_id', 'is_correct'])
    Ytest = list(df_test["is_correct"])
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=1, splitter="best", max_depth=50)
    clf = clf.fit(Xtrain, Ytrain)
    y_train_predict = clf.predict(Xtrain)
    y_test_predict = clf.predict(Xtest)
    # Compute accuracy based on test samples
    train_accuracy = accuracy_score(Ytrain, y_train_predict)
    test_accuracy = accuracy_score(Ytest, y_test_predict)
    print(train_accuracy, test_accuracy)
    clf = LogisticRegression(C=1.0).fit(Xtrain, Ytrain)
    y_train_predict = clf.predict(Xtrain)
    y_test_predict = clf.predict(Xtest)
    train_accuracy = accuracy_score(Ytrain, y_train_predict)
    test_accuracy = accuracy_score(Ytest, y_test_predict)
    print(train_accuracy, test_accuracy)
    clf = SVC(C=1, kernel='rbf', gamma=0.1).fit(Xtrain, Ytrain)
    y_train_predict = clf.predict(Xtrain)
    y_test_predict = clf.predict(Xtest)
    train_accuracy = accuracy_score(Ytrain, y_train_predict)
    test_accuracy = accuracy_score(Ytest, y_test_predict)
    print(train_accuracy, test_accuracy)