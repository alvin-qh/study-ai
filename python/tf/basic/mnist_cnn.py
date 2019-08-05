"""
refer to: http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
"""

import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import conf, log
from dataset.mnist import Mnist, Dataset
from tf.model.saver import Saver

saver = Saver('mnist_cnn')


def _weight_var(shape: Union[tuple, list]) -> tf.Variable:
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W)


def _bias_var(shape: Union[tuple, list]) -> tf.Variable:
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)


def _conv2d(x: tf.Tensor, W: tf.Variable) -> tf.Tensor:
    return tf.nn.conv2d(x, W, strides=(1, 1, 1, 1), padding='SAME')


def _max_pool_2x2(x: tf.Variable) -> tf.Tensor:
    return tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def _pool1(x: tf.Tensor) -> tf.Tensor:
    W_conv1 = _weight_var((5, 5, 1, 32))
    b_conv1 = _bias_var((32,))

    x_image = tf.reshape(x, (-1, 28, 28, 1))

    h_conv1 = tf.nn.relu(_conv2d(x_image, W_conv1) + b_conv1)
    return _max_pool_2x2(h_conv1)


def _pool2(h_pool1: tf.Tensor) -> tf.Tensor:
    W_conv2 = _weight_var((5, 5, 32, 64))
    b_conv2 = _bias_var((64,))

    h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)
    return _max_pool_2x2(h_conv2)


def _full_connect(h_pool2: tf.Tensor) -> tf.Tensor:
    W_fc1 = _weight_var((7 * 7 * 64, 1024))
    b_fc1 = _bias_var((1024,))

    h_pool2_flat = tf.reshape(h_pool2, (-1, 7 * 7 * 64))
    return tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


def _drop_out(h_fc1: tf.Tensor, keep_prob: tf.Tensor) -> tf.Tensor:
    return tf.nn.dropout(h_fc1, keep_prob)


def _softmax(h_fc1_drop: tf.Tensor) -> tf.Tensor:
    W_fc2 = _weight_var((1024, 10))
    b_fc2 = _bias_var((10,))
    return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='softmax')


def _train_step(softmax, y, learning_rate):
    cross_entropy = -tf.reduce_sum(y * tf.log(softmax))
    return tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


def _validator(softmax, y):
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, 'float'))


def train(train_data: Dataset, learning_rate=1e-4, *, train_times=1000, batch_size=100):
    x = tf.placeholder('float', (None, 28 * 28), name='x')
    keep_prob = tf.placeholder('float', name='keep_prob')

    y = tf.placeholder('float', (None, 10))

    h_pool1 = _pool1(x)
    h_pool2 = _pool2(h_pool1)
    h_fc_1 = _full_connect(h_pool2)
    h_fc_1_drop = _drop_out(h_fc_1, keep_prob)
    softmax = _softmax(h_fc_1_drop)
    train_step = _train_step(softmax, y, learning_rate)

    validator = _validator(softmax, y)

    train_accuracy = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, train_times + 1):
            data, labels = train_data.batch(batch_size)
            sess.run(train_step, feed_dict={x: data, y: labels, keep_prob: 0.5})
            if i % 100 == 0:
                train_accuracy = validator.eval(feed_dict={x: data, y: labels, keep_prob: 1.0})
                log.info('step {}, training accuracy {}'.format(i, train_accuracy))

        saver.save(sess)

    return train_accuracy


def validate(data: np.array, labels: np.array):
    with tf.Session() as sess:
        graph = saver.restore(sess)

        x = graph.get_tensor_by_name_0('x')
        keep_prob = graph.get_tensor_by_name_0('keep_prob')
        softmax = graph.get_tensor_by_name_0('softmax')

        y = tf.placeholder('float', (None, 10))

        validator = _validator(softmax, y)

        return sess.run(validator, feed_dict={x: data, y: labels, keep_prob: 1.0})


def predict(data: np.array):
    with tf.Session() as sess:
        graph = saver.restore(sess)

        x = graph.get_tensor_by_name_0('x')
        keep_prob = graph.get_tensor_by_name_0('keep_prob')
        softmax = graph.get_tensor_by_name_0('softmax')

        return sess.run(tf.argmax(softmax, 1), feed_dict={x: data, keep_prob: 1.0})


def main():
    dataset = Mnist()
    train_data = dataset.train_data()
    test_data = dataset.test_data()

    accuracy = train(train_data)
    print('Training accuracy is: {}'.format(accuracy))

    data, labels = test_data.all()
    accuracy = validate(data, labels)
    print('Testing accuracy is: {}'.format(accuracy))

    data, label = test_data[random.randint(0, len(test_data) - 1)]
    result = predict(np.array([data]))
    print('Result is: {}, and the correct label is: {}'.format(result[0], np.argmax(label)))

    if conf.get('SHOW_IMAGE', False):
        print('Show recognized image...')
        data = np.multiply(data, 255.0).astype(np.int)
        plt.imshow(data.reshape((28, 28)), cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
