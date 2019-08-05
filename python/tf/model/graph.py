from typing import Dict, Union

import tensorflow as tf


class Graph:
    """
    计算图对象
    """

    def __init__(self, **kwargs: Dict[str, Union[tf.Session, tf.Graph]]):
        graph = None
        if 'session' in kwargs:
            sess = kwargs.get('session')
            if not isinstance(sess, tf.Session):
                raise TypeError('session is not instanceof \"tf.Session\"')
        elif 'graph' in kwargs:
            graph = kwargs.get('graph')

        self._graph = graph if graph else tf.get_default_graph()

    def get_tensor_by_name_0(self, name: str) -> tf.Tensor:
        """
        从计算图中获取某名称对应的第一个张量
        :param name: 张量名称
        :return: 张量
        """
        return self._graph.get_tensor_by_name('{}:0'.format(name))
