import random
from typing import Tuple

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


class Dataset:
    def __init__(self, data, labels, label_names):
        self._data = data
        self._labels = labels
        self._label_names = label_names
        self._cur = 0

    def find_label_name(self, label) -> str:
        return self._label_names[label]

    def next_batch(self, batch_size=100) -> Tuple[np.array, np.array]:
        total_data, total_labels = np.zeros((0, self._data.shape[1])), np.zeros((0, len(self._label_names)))

        while batch_size > 0:
            cur_to = self._cur + batch_size
            data, labels = self._data[self._cur:cur_to], self._labels[self._cur:cur_to]
            total_data = np.append(total_data, data, axis=0)
            total_labels = np.append(total_labels, labels, axis=0)
            cur_to = self._cur + len(data)
            batch_size -= len(total_data)
            if cur_to == len(self._data):
                self._shuffle()
                self._cur = 0
            else:
                self._cur = cur_to

        return total_data, total_labels

    def _shuffle(self):
        indices = list(range(len(self._data)))
        random.shuffle(indices)
        for i in range(len(indices) // 2):
            self._data[i], self._data[indices[i]] = self._data[indices[i]], self._data[i]
            self._labels[i], self._labels[indices[i]] = self._labels[indices[i]], self._labels[i]

    def __len__(self):
        return len(self._data)

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def position(self) -> int:
        return self._cur

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._data[index], self._labels[index]

    def all(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._data, self._labels


class Iris:
    def __init__(self, one_hot=True, debug=False):
        if debug:
            np.random.seed(1)

        self.dataset = datasets.load_iris()
        self.dataset.data = self._scaler(self.dataset.data)
        if one_hot:
            self.dataset.target = self._one_hot_labels(self.dataset.target, len(self.dataset.target_names))

        self.one_hot = one_hot
        self._train_indices, self._test_indices = self._make_train_and_test_indices(self.dataset.data)

    def __len__(self):
        return len(self.dataset.data)

    @staticmethod
    def _scaler(data):
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler.transform(data)

    @staticmethod
    def _one_hot_labels(labels: np.ndarray, size) -> np.ndarray:
        def _one_hot(n):
            r = [0] * size
            r[n] = 1
            return r

        return np.array([_one_hot(n) for n in labels])

    @staticmethod
    def _make_train_and_test_indices(data):
        length = len(data)
        train_indices = np.random.choice(length, round(length * .8), replace=False)
        test_indices = np.array(list(set(range(length)) - set(train_indices)))
        return train_indices, test_indices

    def train_data(self) -> Dataset:
        return Dataset(self.dataset.data[self._train_indices],
                       self.dataset.target[self._train_indices],
                       self.dataset.target_names)

    def test_data(self) -> Dataset:
        return Dataset(self.dataset.data[self._test_indices],
                       self.dataset.target[self._test_indices],
                       self.dataset.target_names)
