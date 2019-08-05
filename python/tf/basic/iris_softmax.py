import random

import numpy as np
import tensorflow as tf

from common import log
from dataset.iris import Dataset, Iris
from tf.model.saver import Saver

saver = Saver('iris_softmax')


def _softmax(x: tf.Tensor) -> tf.Tensor:
    W = tf.Variable(tf.zeros((4, 3)))
    b = tf.Variable(tf.zeros((3,)))
    return tf.nn.softmax(tf.matmul(x, W) + b, name='softmax')


def _train_step(softmax: tf.Tensor, y: tf.Tensor, learning_rate) -> tf.Operation:
    cross_entropy = -tf.reduce_sum(y * tf.log(softmax))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    return train_step


def _validator(softmax: tf.Tensor, y: tf.Tensor):
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


def train(dataset: Dataset, learning_rate=0.01):
    x = tf.placeholder('float', (None, 4), name='x')
    y = tf.placeholder('float', (None, 3))

    softmax = _softmax(x)

    train_step = _train_step(softmax, y, learning_rate=learning_rate)
    validator = _validator(softmax, y)

    train_accuracy = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            batch_xs, batch_ys = dataset.next_batch(50)
            feed_dict = {x: batch_xs, y: batch_ys}

            sess.run(train_step, feed_dict=feed_dict)
            if i % 100 == 0:
                train_accuracy = validator.eval(feed_dict=feed_dict)
                log.info('step {}, training accuracy {}'.format(i, train_accuracy))

        saver.save(sess)

    return train_accuracy


def validate(data, labels):
    with tf.Session() as sess:
        graph = saver.restore(sess)

        x = graph.get_tensor_by_name_0('x')
        y = tf.placeholder('float', [None, 3])

        softmax = graph.get_tensor_by_name_0('softmax')
        validator = _validator(softmax, y)

        return sess.run(validator, feed_dict={x: data, y: labels})


def predict(data: np.array):
    with tf.Session() as sess:
        graph = saver.restore(sess)

        x = graph.get_tensor_by_name_0('x')
        softmax = graph.get_tensor_by_name_0('softmax')

        labels = sess.run(tf.argmax(softmax, 1), feed_dict={x: data})
        return labels


def main():
    dataset = Iris()
    train_data = dataset.train_data()
    test_data = dataset.test_data()

    accuracy = train(train_data)
    print('Training accuracy is: {}'.format(accuracy))

    data, labels = test_data.all()
    accuracy = validate(data, labels)
    print('Testing accuracy is: {}'.format(accuracy))

    data, label = test_data[random.randint(0, len(test_data) - 1)]
    result = predict(np.array([data]))
    print('Result is: {}, and the correct label is: {}, name is: "{}"'.format(result[0], np.argmax(label),
                                                                              test_data.find_label_name(result[0])))


if __name__ == '__main__':
    main()
