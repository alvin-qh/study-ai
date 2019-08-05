import multiprocessing
from abc import ABCMeta, abstractmethod
from math import sqrt
from threading import RLock

from sklearn import neighbors

from .face_recognize import face_distance, face_compare


class Classifier(metaclass=ABCMeta):
    __classifier_type__ = ''

    @abstractmethod
    def train(self, encodings, labels):
        pass

    @abstractmethod
    def update(self, new_encodings, new_labels, removed_labels):
        pass

    @abstractmethod
    def list_similar(self, encoding, count):
        pass

    @abstractmethod
    def compare(self, encoding, tolerance):
        pass

    @property
    def type(self):
        return self.__classifier_type__

    @staticmethod
    def create(classifier_type):
        if classifier_type == DistanceClassifier.__classifier_type__:
            return DistanceClassifier()
        if classifier_type == KNNClassifier.__classifier_type__:
            return KNNClassifier()

        raise ClassifierError('invalid classifier type {}'.format(classifier_type))


class ClassifierError(Exception):
    pass


_lock = RLock()


class DistanceClassifier(Classifier):
    __classifier_type__ = 'mini_distance'

    def __init__(self):
        self._labels = []
        self._encodings = []

    def train(self, encodings, labels):
        encodings = encodings or []
        labels = labels or []

        if len(encodings) != len(labels):
            raise ClassifierError('len(encodings) != len(labels)')

        with _lock:
            self._labels, self._encodings = labels, encodings

    def update(self, new_encodings, new_labels, removed_labels):
        new_encodings = new_encodings or ()
        new_labels = new_labels or ()

        if len(new_encodings) != len(new_labels):
            raise ClassifierError('len(new_encodings) != len(new_labels)')

        if new_labels or removed_labels:
            with _lock:
                dataset = {label: encoding for label, encoding in zip(self._labels, self._encodings)}
                if new_labels:
                    for label, encoding in zip(new_labels, new_encodings):
                        dataset[label] = encoding

                if removed_labels:
                    for label in removed_labels:
                        dataset.pop(label, None)

                self._labels, self._encodings = zip(*dataset.items())

    def compare(self, encoding, tolerance):
        if not self._labels:
            return None, None

        min_distance, min_distance_index = tolerance, -1
        distances = face_distance(self._encodings, encoding)
        for i, distance in enumerate(distances):
            if distance < min_distance:
                min_distance = distance
                min_distance_index = i

        return (None, None) if min_distance_index < 0 else (self._labels[min_distance_index], min_distance)

    def list_similar(self, encoding, count):
        if not encoding or not self._labels:
            return None

        distances = face_distance(self._encodings, encoding)
        sorted_distance = sorted(enumerate(distances), key=lambda d: d[1])
        return [(d[1], self._labels[d[0]]) for d in sorted_distance[0:count]]


class KNNClassifier(Classifier):
    __classifier_type__ = 'knn'

    def __init__(self):
        self._encoding_dataset = {}
        self._knn_clf = None

    def train(self, encodings, labels):
        encodings = encodings or []
        labels = labels or []

        if len(encodings) != len(labels):
            raise ClassifierError('len(encodings) != len(labels)')

        encoding_dataset = {label: encoding for label, encoding in zip(labels, encodings)}
        with _lock:
            self._knn_clf = self._train(encoding_dataset)
            self._encoding_dataset = encoding_dataset

    @staticmethod
    def _train(encoding_dataset):
        if not encoding_dataset:
            return None

        labels, encodings = zip(*encoding_dataset.items())
        if len(labels) == 1:
            labels += (0,)
            encodings += ([0] * len(encodings[0]),)

        n_neighbors = min(int(round(sqrt(len(encodings)))), 5)
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree',
                                                 weights='distance', n_jobs=multiprocessing.cpu_count())
        knn_clf.fit(encodings, labels)

        return knn_clf

    def update(self, new_encodings, new_labels, removed_labels):
        new_encodings = new_encodings or []
        new_labels = new_labels or []

        if len(new_encodings) != len(new_labels):
            raise ClassifierError('len(new_encodings) != len(new_labels)')

        if new_labels or removed_labels:
            with _lock:
                encoding_dataset = dict(self._encoding_dataset)
                if new_labels:
                    for label, encoding in zip(new_labels, new_encodings):
                        encoding_dataset[label] = encoding

                if removed_labels:
                    for label in removed_labels:
                        encoding_dataset.pop(label, None)

                self._knn_clf = self._train(encoding_dataset)
                self._encoding_dataset = encoding_dataset

    def compare(self, encoding, tolerance):
        if not self._knn_clf:
            return None, None

        labels = self._knn_clf.predict([encoding])
        if not labels:
            return None, None

        found_encoding = self._encoding_dataset.get(labels[0], None)
        if not found_encoding:
            return None, None

        distance = face_compare(found_encoding, encoding)
        if distance >= tolerance:
            return None, None

        return int(labels[0]), distance

    def list_similar(self, encoding, count):
        if not self._encoding_dataset:
            return []

        labels, encodings = zip(*self._encoding_dataset.items())

        distances = face_distance(encodings, encoding)
        sorted_distance = sorted(enumerate(distances), key=lambda d: d[1])

        return [(d[1], labels[d[0]]) for d in sorted_distance[0:count]]
