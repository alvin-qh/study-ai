import os

import tensorflow as tf

from tf.model.graph import Graph


class Saver:
    """
    用于保存训练模型
    """
    __base_dir__ = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.ckpt'))
    __meta_file__ = 'model'

    def __init__(self, name: str):
        """
        :param name: 保存模型的目录名称
        """
        self._model_path = os.path.join(self.__base_dir__, name)
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

    def save(self, sess: tf.Session):
        """
        保存模型
        :param sess: Tensorflow Session 对象
        """
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self._model_path, self.__meta_file__))

    def restore(self, sess: tf.Session):
        """
        恢复模型
        :param sess: Tensorflow Session 对象
        :return: 恢复的计算图对象
        """
        meta_graph = tf.train.import_meta_graph(
            os.path.join(self._model_path, '{}.{}'.format(self.__meta_file__, 'meta')))
        meta_graph.restore(sess, tf.train.latest_checkpoint(self._model_path))
        return Graph(session=sess)


