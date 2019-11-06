import random
from os import curdir, path, makedirs

import numpy as np
from tensorflow import keras


class Dataset:
    def __init__(self, dataset, one_hot=True):
        self._data, self._labels = self._transcode(dataset, one_hot)
        self._cur = 0

    @staticmethod
    def _transcode(dataset, one_hot: bool):
        data, labels = dataset

        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        data = np.multiply(data.astype(np.float32), 1.0 / 255.0)

        if one_hot:
            def _one_hot(n: int) -> np.ndarray:
                array = [0] * 10
                array[n] = 1
                return np.array(array)

            labels = np.array([_one_hot(l) for l in labels])
        else:
            labels = labels.reshape(labels.shape[0], 1)

        return data, labels

    def batch(self, batch_size=100):
        total_data, total_labels = np.zeros((0, self._data.shape[1])), np.zeros((0, 10))

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
    def size(self):
        return len(self._data)

    @property
    def position(self):
        return self._cur

    def __getitem__(self, index):
        return self._data[index], self._labels[index]

    def all(self):
        return self._data, self._labels


class Mnist:
    _cache_dir = ('.mnist-data', 'mnist.npz')

    def __init__(self):
        data_path = self._create_data_path()
        self._train_data, self._test_data = keras.datasets.mnist.load_data(data_path)

    @classmethod
    def _create_data_path(cls):
        dir_ = path.abspath(path.join(curdir, cls._cache_dir[0]))
        makedirs(dir_, exist_ok=True)
        return path.join(dir_, cls._cache_dir[1])

    def train_data(self):
        return Dataset(self._train_data)

    def test_data(self):
        return Dataset(self._test_data)
