from IPython.display import Image
from six import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import math
import pydotplus

criterion_map = {
    'entropy': 'information gain',
    'gini': 'Gini coefficient'
}


def load_data():
    """
    splits the entire dataset randomly into 70% training, 15% validation, and 15% test examples
    """
    file_fake = []
    with open('clean_fake.txt') as f:
        file_fake = f.read().splitlines()
    file_real = []
    with open('clean_real.txt') as f:
        file_real = f.read().splitlines()
    # combined the fake and real news
    data = file_fake + file_real
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    # vectorize the data
    data_vectorize = vectorizer.transform(data).toarray()
    feature_names = list(vectorizer.get_feature_names())
    label = [0] * len(file_fake) + [1] * len(file_real)
    data_train, data_test, label_train, label_test = train_test_split(data_vectorize, label, test_size=0.3,
                                                                      random_state=1)
    data_valid, data_test, label_valid, label_test = train_test_split(data_test, label_test, test_size=0.5,
                                                                      random_state=1)
    return feature_names, data_train, label_train, data_valid, label_valid, data_test, label_test


def select_model(feature_names, data_train, label_train, data_valid, label_valid, data_test, label_test):
    """
    get the best model using different criterion_options and depth_options also plot tree for the best accuracy
    """
    depth_options = [5, 10, 50, 100, 150]
    criterion_options = ['entropy', 'gini']
    best_acurracy = 0
    best_tree = ''
    for depth_option in depth_options:
        for criterion_option in criterion_options:
            clf = DecisionTreeClassifier(criterion=criterion_option, max_depth=depth_option, random_state=100)
            clf.fit(data_train, label_train)
            label_pred = clf.predict(data_valid)
            accuracy = get_accuracy(label_pred, label_valid)
            if accuracy > best_acurracy:
                best_tree = clf
                best_acurracy = accuracy
            print(
                f'decision tree classifier using max_depth {depth_option} and split criteria {criterion_map[criterion_option]} has accuracy {accuracy}')
    get_plot(best_tree, feature_names)


def get_accuracy(label_pred, label_valid):
    """
    calculate the accuracy on validation set
    """
    rows = len(label_pred)
    match = 0
    for i in range(rows):
        if label_pred[i] == label_valid[i]:
            match += 1
    return match / rows


def get_plot(clf, feature_names):
    """
    get the best tree diagram from clf and use feature_names
    """
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, max_depth=2, rounded=True,
                    special_characters=True, feature_names=feature_names, class_names=['fake', 'real'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('best_tree.png')


def compute_information_gain(data, label, keyword, feature_names):
    """
    calculate the information gain
    """
    exist_fake = 0
    exist_real = 0
    not_exist_fake = 0
    not_exist_real = 0
    # num of real before the split
    reals = sum(label)
    # get the keyword index in the feature lst
    keyword_index = list(feature_names).index(keyword)
    # split labels into two set one contains keyword, one does not
    for i in range(len(label)):
        input_ = data[i]
        input_label = label[i]
        if input_[keyword_index] != 0:
            # input contains keyword
            if input_label == 0:
                exist_fake += 1
            else:
                exist_real += 1
        else:
            # input does not contains keyword
            if input_label == 0:
                not_exist_fake += 1
            else:
                not_exist_real += 1
    # compute information gain
    info_gain = entropy(len(label) - reals, reals) - ((exist_real + exist_fake) / len(label)) * entropy(exist_fake,
                                                                                                      exist_real) - (
            (not_exist_real + not_exist_fake) / len(label)) * entropy(not_exist_fake, not_exist_real)
    return info_gain


def entropy(fake_count, real_count):
    """
    calculaye the entropy
    """
    prob = real_count / (fake_count + real_count)
    prob_neg = 1 - prob
    return -prob * math.log(prob, 2) - prob_neg * math.log(prob_neg, 2)


if __name__ == '__main__':
    feature_names, data_train, label_train, data_valid, label_valid, data_test, label_test = load_data()
    select_model(feature_names, data_train, label_train, data_valid, label_valid, data_test, label_test)
    print(
        f'information gain at root split donald is {compute_information_gain(data_train, label_train, "donald", feature_names)}')
    print(
        f'information gain at other keyword the is {compute_information_gain(data_train, label_train, "the", feature_names)}')
    print(
        f'information gain at other keyword hillary is {compute_information_gain(data_train, label_train, "hillary", feature_names)}')
    print(
        f'information gain at other keyword trumps is {compute_information_gain(data_train, label_train, "trumps", feature_names)}')


