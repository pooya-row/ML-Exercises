import numpy as np
from sortedcontainers import SortedList
from util import get_data


class KNN:
    def __init__(self, k):
        self.k = k


if __name__ == '__main__':
    X, Y = get_data(2000)
    print(X[1, :])
