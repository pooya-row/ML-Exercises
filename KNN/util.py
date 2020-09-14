import numpy as np
import pandas as pd


def get_data(limit=None):
    df = pd.read_csv('mnist_train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    x = data[:, 1:] / 255
    y = data[:, 0]
    if limit is not None:
        x, y = x[:limit], y[:, limit]
    return x, y
