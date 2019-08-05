import os
import random

import numpy as np
import tensorflow as tf

from common import log
from dataset.iris import Iris

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.ckpt/iris_dnn_classifier'))


def _create_classifier():
    columns = [tf.feature_column.numeric_column('x', (4,))]
    # feature_columns: 表示特征的字段数
    # model_dir: 存储模型的路径
    # optimizer: 用于定义训练时的优化器。包括：Adagrad, Adam, Ftrl, Momentum, RMSProp, SGD，默认为 `Adagrad`
    # activation_fn：神经元激活函数，默认为 `relu`
    # gradient_clip_norm：梯度下降速率
    # hidden_units：每层网络上 hidden 单元的数量
    # n_classes: 分类个数
    # dropout：退出参数
    # input_layer_min_slice_size：输入层最小切片大小，默认 64M
    return tf.estimator.DNNClassifier(hidden_units=[10, 20, 10],
                                      feature_columns=columns,
                                      model_dir=MODEL_PATH,
                                      n_classes=3,
                                      optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                      activation_fn='relu')


def _validator(classifier, data, labels):
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': data}, y=labels, num_epochs=1, shuffle=False)
    train_accuracy = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    return train_accuracy


def _train_step(classifier, data, labels, steps):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': data}, y=labels, num_epochs=None, shuffle=True)
    classifier.train(input_fn=train_input_fn)


def train(dataset, batch_size=50, train_times=200, train_step=10):
    classifier = _create_classifier()

    train_accuracy = 0
    for i in range(1, train_times + 1):
        data, labels = dataset.next_batch(batch_size)
        labels = np.array(labels, dtype=np.int32)

        _train_step(classifier, data, labels, train_step)
        train_accuracy = _validator(classifier, data, labels)
        print(i)
        if i % 100 == 0:
            log.info('step {}, training accuracy {}'.format(i, train_accuracy))

    return train_accuracy


def validate(data, labels: np.ndarray):
    classifier = _create_classifier()

    input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': data}, num_epochs=1, shuffle=False)
    results = classifier.predict(input_fn=input_fn)

    return np.mean([1 if r else 0 for r in labels == results])


def predict(data: np.array):
    classifier = _create_classifier()
    input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': data}, num_epochs=1, shuffle=False)
    return classifier.predict(input_fn=input_fn)


def main():
    dataset = Iris(one_hot=False)
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
