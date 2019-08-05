import tensorflow as tf

from tf.model.saver import Saver

saver = Saver('tf_session')


def train(name: str):
    # 定义一个存储字符串变量的张量变量
    h = tf.Variable('Hello ', dtype=tf.string)

    # 定义一个名为'w'的占位符张量，存储输入值
    w = tf.placeholder(tf.string, name='w')

    # 进行一次加法操作
    hw = tf.add(h, w, name='op')  # hw = h + w

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(hw, feed_dict={w: name})

        # 存储计算图和权重
        saver.save(sess)

    return result


def predict(name: str):
    with tf.Session() as sess:
        # 恢复计算图和权重
        graph = saver.restore(sess)

        # 恢复存储的张量
        w = graph.get_tensor_by_name_0('w')
        hw = graph.get_tensor_by_name_0('op')
        result = sess.run(hw, feed_dict={w: name})

    return result


def main():
    result = train('World')
    print('Result is: {}'.format(result.decode()))

    result = predict('World1')
    print('Result is: {}'.format(result.decode()))


if __name__ == '__main__':
    main()
