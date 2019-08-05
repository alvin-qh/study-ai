from typing import Dict

import numpy as np

from dataset.mnist import Dataset


def _count_array(array: np.ndarray, one_hot=False) -> Dict[int, int]:
    result = {}
    for n in array:
        if one_hot:
            key = np.argmax(n)
        else:
            key = int(n[0])

        result[key] = result.get(key, 0) + 1

    return result


def test_dataset():
    total_data, total_labels = np.zeros((0, 28 * 28)), np.zeros((0, 10))

    data = np.array([[[i * 255.0] * 28] * 28 for i in range(1, 10)])
    labels = np.array([i for i in range(1, 10)])

    ds = Dataset((data, labels))
    assert ds.size == 9
    assert ds.position == 0

    data, labels = ds.batch(5)
    assert ds.size == 9
    assert ds.position == 5

    total_data = np.append(total_data, data, axis=0)
    total_labels = np.append(total_labels, labels, axis=0)

    data, labels = ds.batch(4)
    assert ds.size == 9
    assert ds.position == 0

    total_data = np.append(total_data, data, axis=0)
    total_labels = np.append(total_labels, labels, axis=0)
    assert len(total_data) == 9
    assert len(total_labels) == 9

    counter = _count_array(total_data)
    for i in range(1, 10):
        assert counter[i] == 1

    counter = _count_array(total_labels, one_hot=True)
    for i in range(1, 10):
        assert counter[i] == 1


def test_dataset_overflow():
    data = np.array([[[i * 255.0] * 28] * 28 for i in range(1, 10)])
    labels = np.array([i for i in range(1, 10)])

    ds = Dataset((data, labels))
    ds.batch(5)
    ds.batch(4)

    assert ds.size == 9
    assert ds.position == 0

    ds.batch(8)
    assert ds.size == 9
    assert ds.position == 8

    ds.batch(11)
    assert ds.size == 9
    assert ds.position == 0
