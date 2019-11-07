from os import path, makedirs

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_weights(model: tf.keras.Model, model_path: str, model_file='model') -> bool:
    model_path = path.abspath(path.join(path.dirname(__file__), model_path))
    if not path.exists(model_path):
        return False

    model_path = path.join(model_path, model_file)
    if not path.exists('{}.index'.format(model_path)):
        return False

    model.load_weights(model_path)
    return True


def save_weights(model: tf.keras.Model, model_path: str, model_file='model') -> None:
    model_path = path.abspath(path.join(path.dirname(__file__), model_path))
    if not path.exists(model_path):
        makedirs(model_path, exist_ok=True)

    model_path = path.join(model_path, model_file)
    model.save_weights(model_path)


def show_image(img: np.ndarray, *, figsize=(1, 1), cmap='gray') -> None:
    plt.figure(figsize=figsize)
    if len(img.shape) > 2:
        img = img.reshape(img.shape[1:])

    plt.imshow(img, cmap=cmap)
    plt.show()
