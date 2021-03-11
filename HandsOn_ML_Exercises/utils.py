from typing import List, Tuple

import numpy as np


def scale_data(x: np.array) -> np.array:
    """
    Scale data between 0 to 1.
    :param x: numpy array to be normalized
    :return: normalized result of x
    """
    return x / 255.


def reduce_date(x: np.array, y: np.array, n: int =5120) -> Tuple[np.array, np.array]:
    """

    :param x: numpy array containing features of all observations
    :param y: numpy array containing labels for all observations
    :param n: int, number of desired observations
    :return: tuple containing the reduced x and y
    """
    ix = np.random.randint(0, x.shape[0], n)
    return x[ix], y[ix]
