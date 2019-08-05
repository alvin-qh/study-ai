import random

import numpy as np
from sklearn import svm, neighbors, linear_model

from dataset.country import CountryStatsDataset
from dataset.iris import Iris

iris = Iris(one_hot=False)


def linear():
    dataset = CountryStatsDataset()

    classifier = linear_model.LinearRegression()
    # 设定 GDP 和 生活指数 的对应关系
    classifier.fit(dataset.gdp_per_capita, dataset.life_satisfaction)

    new_gdp = [[22587]]
    new_life_satisfaction = classifier.predict(X=new_gdp)
    print('GDP {} may got life satisfaction {}'.format(new_gdp[0], new_life_satisfaction[0]))


def svm_svc_1():
    train_dataset = iris.train_data()
    test_dataset = iris.test_data()

    train_data, train_labels = train_dataset.all()
    classifier = svm.LinearSVC(multi_class='ovr', random_state=0, C=1)
    classifier.fit(train_data, train_labels)

    test_data, test_labels = test_dataset.all()
    results = classifier.predict(test_data)
    correct_prediction = [1 if n else 0 for n in test_labels == results]
    print('Correct prediction is: {}'.format(np.mean(correct_prediction)))

    test_data, test_label = test_dataset[random.randint(0, len(test_data) - 1)]
    results = classifier.predict([test_data])
    print('Result is: {} and expected is: {}, name is: "{}"'.format(results[0], test_label,
                                                                    test_dataset.find_label_name(results[0])))


def svm_svc_2():
    train_dataset = iris.train_data()
    test_dataset = iris.test_data()

    train_data, train_labels = train_dataset.all()
    classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
    classifier.fit(train_data, train_labels)

    test_data, test_labels = test_dataset.all()
    results = classifier.predict(test_data)
    correct_prediction = [1 if n else 0 for n in test_labels == results]
    print('Correct prediction is: {}'.format(np.mean(correct_prediction)))

    test_data, test_label = test_dataset[random.randint(0, len(test_data) - 1)]
    results = classifier.predict([test_data])
    print('Result is: {} and expected is: {}, name is: "{}"'.format(results[0], test_label,
                                                                    test_dataset.find_label_name(results[0])))


def knn():
    train_dataset = iris.train_data()
    test_dataset = iris.test_data()

    train_data, train_labels = train_dataset.all()
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', metric='euclidean')
    classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree', metric='minkowski')
    classifier.fit(train_data, train_labels)
    print('KNN score is: {}'.format(classifier.score(train_data, train_labels)))

    test_data, test_labels = test_dataset.all()
    results = classifier.predict(test_data)
    correct_prediction = [1 if n else 0 for n in test_labels == results]
    print('Correct prediction is: {}'.format(np.mean(correct_prediction)))

    test_data, test_label = test_dataset[random.randint(0, len(test_data) - 1)]
    results = classifier.predict([test_data])
    print('Result is: {} and expected is: {}, name is: "{}"'.format(results[0], test_label,
                                                                    test_dataset.find_label_name(results[0])))


def main():
    linear()
    print()

    svm_svc_1()
    print()

    svm_svc_2()
    print()

    knn()
    print()


if __name__ == '__main__':
    main()
