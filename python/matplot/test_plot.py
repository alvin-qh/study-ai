import matplotlib.pyplot as plt


def test_linear_image_dim_1():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('Numbers')
    plt.show()


def test_linear_image_dim_2():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.ylabel('Numbers')
    plt.axis([0, 6, 0, 20])  # [min_x, max_x, min_y, max_y]
    plt.show()


def main():
    test_linear_image_dim_1()
    print()

    test_linear_image_dim_2()


if __name__ == '__main__':
    main()
