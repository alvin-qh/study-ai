import numpy as np


def _format(array: np.ndarray):
    return np.around(array, decimals=2).tolist()


def add():
    """
    m * n
    """
    a = 10.0
    b = 0.1
    print('{} + {} = {}'.format(a, b, _format(np.add(a, b))))

    """
    A(2,3) + n
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = 0.1
    print('{} + {} = {}'.format(a, b, _format(np.add(a, b))))
    #
    """
    A(2,3) + B(1,3)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [0.1, 0.2, 0.3]
    print('{} + {} = {}'.format(a, b, _format(np.add(a, b))))

    """
    A(2,3) + B(1, 4)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [0.1, 0.2, 0.3, 0.4]
    try:
        np.add(a, b)
    except Exception as err:
        print('{} + {} cause error: {}'.format(a, b, err))

    """
    A(2,3) + B(2, 2)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2], [0.4, 0.5]]
    try:
        np.add(a, b)
    except Exception as err:
        print('{} + {} cause error: {}'.format(a, b, err))

    """
    A(2,3) + B(2, 3)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    print('{} + {} = {}'.format(a, b, _format(np.add(a, b))))


def multiply():
    """
    tf.multiply does element-wise multiplication
        点乘计算要求两个矩阵拥有相同的维度，相同下标的元素一一对应相乘，比如 mn 维矩阵只能和 mn 维矩阵相乘，
    获得的结果还是 mn 维矩阵。

    tf.matmul does matrix multiplication
        两个矩阵做乘法，必须满足第一个矩阵的第二维和第二个矩阵的第一维要相等，结果的维度等于第一个矩阵的第一
    维和第二个矩阵的第二维。 比如 mn 的矩阵和 np 的矩阵，结果为 mp 维矩阵。计算过程是第一个矩阵的每一行和
    第二个矩阵的每一列，即两个向量相乘。
    """

    """
    a • b
    """
    a = float(10.0)
    b = float(0.1)
    print('{} • {} = {}'.format(a, b, _format(np.multiply(a, b))))

    """
    A(2,3) • n
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = 0.1
    print('{} • {} = {}'.format(a, b, _format(np.multiply(a, b))))

    """
    A(2,3) • B(1,3)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2, 0.3]]
    print('{} • {} = {}'.format(a, b, _format(np.multiply(a, b))))

    """
    A(2,3) • B(1,4)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2, 0.3, 0.4]]
    try:
        np.multiply(a, b)
    except ValueError as err:
        print(err)

    """
    A(2,3) * B(1,4)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2, 0.3, 0.4]]
    try:
        np.matmul(a, b)
    except ValueError as err:
        print(err)

    """
    A(2,3) * B(3,2)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    print('{} * {} = {}'.format(a, b, _format(np.matmul(a, b))))

    """
    A(2,3) * B(3,2) != B(3,2) * A(2,3)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    print('{} * {} = {} is not eq than {} * {} = {}'.format(
        a, b, _format(np.matmul(a, b)),
        b, a, _format(np.matmul(b, a))))

    """
    A(2,3) * B(3,2) * C(2,2)
    
    A(2,3) * B(3,2) is AB(2,2)
    AB(2,2) * C(2,2) is ABC(2,2)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    c = [[10, 20], [30, 40]]
    print('{} * {} * {} = {}'.format(a, b, c, _format(np.matmul(np.matmul(a, b), c))))

    """
    A(2,3) * (B(3,2) * C(2,2)) = (A(2,3) * B(3,2)) * C(2,2)
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    c = [[10, 20], [30, 40]]
    print('({} * {}) * {} = {} is eq than {} * ({} * {}) = {}'.format(
        a, b, c, _format(np.matmul(np.matmul(a, b), c)),
        a, b, c, _format(np.matmul(a, np.matmul(b, c)))))

    """
    (A(2,2) + B(2,2)) * C(2,2) == (A(2,2) * C(2,2)) + (B(2,2) * C(2,2))
    """
    a = [[1, 2], [3, 4]]
    b = [[0.1, 0.2], [0.3, 0.4]]
    c = [[10, 20], [30, 40]]
    print('({} + {}) * {} = {} is eq than ({} * {}) + ({} * {}) = {}'.format(
        a, b, c, _format(np.matmul(np.add(a, b), c)),
        a, c, b, c, _format(np.add(np.matmul(a, c), np.matmul(b, c)))
    ))


def matrix_opt():
    """
    Random in array
    """
    a = [n for n in range(1, 11)]
    ar = np.random.choice(a, len(a), replace=False)
    print('{} is random as {}'.format(a, _format(ar)))

    """
    Matrix with x and y
    """
    a = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
    # 获取第一行
    print('First row of {} is {}'.format(_format(a), _format(a[0])))

    # 获取第 1 列
    print('First column of {} is {}'.format(_format(a), _format(a[:, 0])))

    # 获取最后 1 列
    print('Last row of {} is {}'.format(_format(a), _format(a[:, -1])))

    # 获取最后 1 列的后 2 行
    print('Last column and last 2 rows of {} is {}'.format(_format(a), _format(a[1:, -1])))

    #
    print('First 2 column of {} is {}'.format(_format(a), _format(a[:, :2])))


def reshape():
    """
    reshape, 将矩阵从一种结构转换为另一种结构，例如：
    reshape(A(3 * 2), (2, 3)) 得到一个 A'(2 * 3) 的矩阵，矩阵内数值排列不变
    """

    a = np.array([1, 2, 3, 4, 5, 6])
    print('Shape of {} is {}'.format(_format(a), a.shape))

    b = np.reshape(a, (1, 6))  # 1 * 6 = 6
    print('{} reshape to {} is {}'.format(_format(a), b.shape, _format(b)))

    b = np.reshape(a, (2, 3))  # 2 * 3 = 6
    print('{} reshape to {} is {}'.format(_format(a), b.shape, _format(b)))

    b = np.reshape(a, (3, 2))  # 3 * 2 = 6
    print('{} reshape to {} is {}'.format(_format(a), b.shape, _format(b)))

    b = np.reshape(a, (6, 1))  # 6 * 1 = 6
    print('{} reshape to {} is {}'.format(_format(a), b.shape, _format(b)))

    a = np.array([[1], [2], [3], [4], [5], [6]])
    print('Shape of {} is {}'.format(_format(a), a.shape))

    b = np.reshape(a, (3, 2))
    print('{} reshape to {} is {}'.format(_format(a), b.shape, _format(b)))

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    print('Shape of {} is {}'.format(_format(a), a.shape))

    b = np.reshape(a, (2, 3, 2))  # 2 * 3 * 2 = 12
    print('{} reshape to {} is {}'.format(_format(a), b.shape, _format(b)))

    a = np.array([[11, 12, 13, 14],
                  [21, 22, 23, 24],
                  [31, 32, 33, 34],
                  [41, 42, 43, 44],
                  [51, 52, 53, 54]])
    print('Shape of {} is {}'.format(_format(a), a.shape))

    b = np.reshape(a, (-1, 2, 2, 1))  # -1 表示第 1 维和原矩阵相同
    print('{} reshape to {} is {}'.format(_format(a), b.shape, _format(b)))


# noinspection PyPep8Naming
def T_transpose():
    """
    轴对称矩阵转置
    将矩阵转置90度，例如：
    T(A(3 * 2)) 得到一个 A(2 * 3) 的矩阵
    """

    a = np.array([[1, 2, 3], [4, 5, 6]])
    # A(2 * 3) => A'(3 * 2)
    # A(0, 0)  => A'(0, 0)
    # A(0, 1)  => A'(1, 0)
    # A(0, 2)  => A'(2, 0)
    # A(1, 0)  => A'(0, 1)
    # A(1, 1)  => A'(1, 1)
    # A(1, 2)  => A'(2, 1)
    # =>
    # [[1 2 3]     [[1 4]
    #  [4 5 6]] =>  [2 5]
    #               [3 6]]
    print('T of {} is {}'.format(_format(a), _format(a.T)))

    a = np.array([[1, 2], [3, 4], [5, 6]])
    # A(3 * 2) => A'(2 * 3)
    # A(0, 0)  => A'(0, 0)
    # A(0, 1)  => A'(1, 0)
    # A(1, 0)  => A'(0, 1)
    # A(1, 1)  => A'(1, 1)
    # A(2, 0)  => A'(0, 2)
    # A(2, 1)  => A'(1, 2)
    # =>
    # [[1 2]       [[1 3 5]
    #  [3 4]   =>   [2 4 6]]
    #  [5 6]]
    print('T of {} is {}'.format(_format(a), _format(a.T)))

    a = np.array([[[11, 12, 13], [14, 15, 16]]])
    # A(1 * 2 * 3) => A'(3 * 2 * 1)
    # A(0, 0, 0) => A'(0, 0, 0)
    # A(0, 0, 1) => A'(1, 0, 0)
    # A(0, 0, 2) => A'(2, 0, 0)
    # A(0, 1, 0) => A'(0, 1, 0)
    # A(0, 1, 1) => A'(1, 1, 1)
    # A(0, 1, 2) => A'(2, 0, 0)
    # =>
    # [[[11 12 13]          [[[11]
    #   [14, 15, 16]]]  =>    [14]]
    #                        [[12]
    #                         [15]]
    #                        [[13]
    #                         [16]]]
    print('T of {} is {}'.format(_format(a), _format(a.T)))


def transpose():
    """
    按指定规则转置
    """

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print('Shape of {} is {}'.format(_format(a), a.shape))

    # 默认的转置方法相当于轴对称转置, 即 a.T
    print('Default transpose of {} is {}'.format(_format(a), _format(np.transpose(a))))

    # 将 shape 为 (2, 3) 的矩阵按 (1, 0) 的规则进行转置，即转置为 (3, 2), 仍为轴对称转置
    print("[1, 0] transpose of {} is {}".format(_format(a), _format(np.transpose(a, [1, 0]))))

    a = np.reshape(range(1, 25), (4, 3, 2))
    print('Shape of {} is {}'.format(_format(a), a.shape))

    # 将 shape 为 (4, 3, 2) 的矩阵按 (1, 2, 0) 的规则转置为 (3, 2, 4)
    print("[1, 2, 0] transpose of {} is {}".format(_format(a), _format(np.transpose(a, [1, 2, 0]))))


def main():
    add()
    print()

    multiply()
    print()

    matrix_opt()
    print()

    reshape()
    print()

    T_transpose()
    print()

    transpose()


if __name__ == '__main__':
    main()
