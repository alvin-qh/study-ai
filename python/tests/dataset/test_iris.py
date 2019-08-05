from dataset.iris import Iris


def test_load_data():
    iris = Iris()
    train_data = iris.train_data()
    test_data = iris.test_data()

    assert len(iris) == len(train_data) + len(test_data)
    assert len(train_data) / len(iris) == 0.8
