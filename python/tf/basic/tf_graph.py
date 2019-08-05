import numpy as np
import tensorflow as tf


def constraint():
    v0 = 3
    c0 = tf.constant(v0, dtype=tf.int32)

    v1 = [3., 4.1, 5.2]
    c1 = tf.constant(v1, dtype=tf.float64)

    v2 = [['Apple', 'Orange'], ['Potato', 'Tomato']]
    c2 = tf.constant(v2, dtype=tf.string)

    v3 = [[[5], [6], [7]], [[4], [3], [2]]]
    c3 = tf.constant(v3, dtype=tf.int32)

    c4_1 = tf.constant([[1, 2, 3], [4, 5, 6]])
    c4_2 = tf.constant([[1, 4], [2, 5], [3, 6]])
    product = tf.matmul(c4_1, c4_2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Value is: {} and tf constraint is: {}'.format(v0, sess.run(c0)))
        print('Value is: {} and tf constraint is: {}'.format(v1, sess.run(c1).tolist()))
        print('Value is: {} and tf constraint is: {}'.format(v2, sess.run(c2).tolist()))
        print('Value is: {} and tf constraint is: {}'.format(v3, sess.run(c3).tolist()))
        print('Value ({}) * ({}) is: {}'.format(sess.run(c4_1).tolist(),
                                                sess.run(c4_2).tolist(),
                                                sess.run(product).tolist()))


def variable_tensor():
    v1_1 = tf.Variable(10)
    v1_2 = tf.Variable(20)
    op1 = tf.assign(v1_1, tf.add(v1_1, v1_2))

    v2_1 = tf.Variable(10)
    v2_2 = tf.placeholder(tf.int32)
    op2 = tf.assign(v2_1, tf.add(v2_1, v2_2))

    # 产生随机数矩阵，标准差为2
    v3_1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0, dtype=tf.float64))

    # 产生随机数矩阵，平均值为1.5
    v3_2 = tf.Variable(tf.random_normal([2, 3], mean=2.0, dtype=tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        n1 = sess.run(v1_1)
        n2 = sess.run(v1_2)
        sess.run(op1)
        print('Operation: ({} + {} = {})'.format(n1, n2, sess.run(v1_1)))

        n1 = sess.run(v1_1)
        n2 = sess.run(v1_2)
        sess.run(op1)
        print('Operation: ({} + {} = {})'.format(n1, n2, sess.run(v1_1)))

        n1 = sess.run(v2_1)
        n2 = 20
        sess.run(op2, feed_dict={v2_2: n2})
        print('Operation: ({} + {} = {})'.format(n1, n2, sess.run(v2_1)))

        n1 = sess.run(v3_1)
        n2 = sess.run(v3_2)
        print('For {}\n\tShape is: {} and standard deviation is {}'.format(n1.tolist(),
                                                                           n1.shape,
                                                                           np.std(n1)))
        print('For {}\n\tShape is: {} and mean deviation is {}'.format(n2.tolist(),
                                                                       n2.shape,
                                                                       np.mean(n2)))


def positive_communication():
    w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
    b1 = tf.Variable(tf.constant(0.0, shape=(3,)))

    w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
    b2 = tf.Variable(tf.constant(0.0, shape=(1,)))

    # 输入张量, 2 x 2
    input_ = tf.constant([[0.2, 0.4], [0.7, 0.9]])

    # ? x 2 -> ? x 3
    relu_1 = tf.nn.relu(tf.matmul(input_, w1) + b1)
    # ? x 3 -> ? x 1
    relu_2 = tf.nn.relu(tf.matmul(relu_1, w2) + b2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        input_ = sess.run(input_)
        print('Input tensor is: {}, shape is: {}'.format(input_.tolist(), input_.shape))

        relu_1 = sess.run(relu_1)
        print('Relu 1 tensor is: {}, shape is: {}'.format(relu_1.tolist(), relu_1.shape))

        relu_2 = sess.run(relu_2)
        print('Relu 2 tensor is: {}, shape is: {}'.format(relu_2.tolist(), relu_2.shape))


def truncated_normal():
    # 正态分布随机数, 标准差0.1
    x = tf.Variable(tf.truncated_normal((5, 5, 1, 32), stddev=0.1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        x = sess.run(x)

    print('Truncated normal result is: [shape={}, std={}]'.format(x.shape, np.std(x)))


def main():
    constraint()
    print()

    variable_tensor()
    print()

    positive_communication()
    print()

    truncated_normal()


if __name__ == '__main__':
    main()
