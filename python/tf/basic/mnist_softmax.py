"""
refer to: http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import log, conf
from dataset.mnist import Mnist, Dataset
from tf.model.saver import Saver

saver = Saver('mnist_softmax')


def _create_softmax(x: tf.Tensor) -> tf.Tensor:
    """
    定义一个 softmax 回归
    softmax 回归一般做为输出层，输出各个类别的概率，做为神经网络最终的输出结果，可以根据概率进行分类。

        evidence(i) = ∑(0~j) W(i,j) * x(j) + b(i)

    j 代表给定图片 x 的像素索引用于像素求和。然后用 softmax 函数可以把这些证据转换成概率 y

        y = softmax(evidence)

    这里的 softmax 可以看成是一个激励(activation)函数或者链接(link)函数，把我们定义的线性函数的输出转换成我们想要的格式，
    也就是关于10个数字类的概率分布。

    因此，给定一张图片，它对于每一个数字的吻合度可以被 softmax 函数转换成为一个概率值。
    softmax函数可以定义为：

        softmax(x) = normalize(exp(x))
    """
    # W(i) 表示结果i的权重，即输入图像的每个像素和正确结果对应的概率关系
    W = tf.Variable(tf.zeros((28 * 28, 10)))

    # b(i) 表示结果i的偏置
    b = tf.Variable(tf.zeros((10,)))

    return tf.nn.softmax(tf.matmul(x, W) + b, name='softmax')


def _train_step(softmax: tf.Tensor, y: tf.Tensor, learning_rate: float):
    """
    定义梯度下降模型，根据神经网络的预测值和真实值比较(交叉熵)，按照给定的学习速率，最小化交叉熵，通过反向传播
    不断的修正softmax中的权重，以期望预测值更符合真实值

    交叉熵定义为

        -(∑ y * log(softmax))

    交叉熵表示一组真实值和一组预测值之间的差值，用于描述预测和真相之间的“损失情况”，用于衡量预测值和真实值的相似度
    """
    # 计算交叉熵
    cross_entropy = -tf.reduce_sum(y * tf.log(softmax))
    # GradientDescentOptimizer表示一个梯度下降优化器
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


def _validator(softmax: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    校验模型
    用于计算预测结果符合真实结果的比例

        tf.argmax(softmax, 1) 表示分类器计算结果
        tf.argmax(y, 1) 表示实际结果
        tf.argmax 用于获取 one hot 数据中 1 的位置，即数据所代表的数字

        tf.reduce_mean 表示取所有符合结果的平均值
    """
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1)), "float"))


def train(train_data: Dataset, learning_rate=0.01, *, train_times=1000, batch_size=100) -> float:
    # 占位符，表示输入的训练数据或待识别数据
    x = tf.placeholder('float', (None, 28 * 28), name='x')

    # 占位符，表示输入的测试数据(用于和 x 输入的数据进行对比)
    y = tf.placeholder('float', (None, 10))

    # 创建回归模型
    softmax = _create_softmax(x)

    # 创建梯度下降模型，训练回归模型
    train_step = _train_step(softmax, y, learning_rate)

    # 创建校验模型
    validator = _validator(softmax, y)

    train_accuracy = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, train_times + 1):
            data, labels = train_data.batch(batch_size)
            feed_dict = {x: data, y: labels}

            sess.run(train_step, feed_dict=feed_dict)
            if i % 100 == 0:
                train_accuracy = validator.eval(feed_dict=feed_dict)
                log.info('step {}, training accuracy {}'.format(i, train_accuracy))

        saver.save(sess)

    return train_accuracy


def validate(data: np.array, labels: np.array):
    with tf.Session() as sess:
        graph = saver.restore(sess)

        x = graph.get_tensor_by_name_0('x')
        y = tf.placeholder('float', (None, 10))
        softmax = graph.get_tensor_by_name_0('softmax')

        validator = _validator(softmax, y)
        return sess.run(validator, feed_dict={x: data, y: labels})


def predict(data: np.array):
    with tf.Session() as sess:
        graph = saver.restore(sess)

        x = graph.get_tensor_by_name_0('x')
        softmax = graph.get_tensor_by_name_0('softmax')

        return sess.run(tf.argmax(softmax, 1), feed_dict={x: data})


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
