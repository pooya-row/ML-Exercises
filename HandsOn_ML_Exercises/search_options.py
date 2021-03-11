from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import idx2numpy
from pprint import pprint
import gzip

from utils import reduce_date, scale_data

IMPORTED_DATA_FILES = ['train-images-idx3-ubyte.gz',
                       'train-labels-idx1-ubyte.gz',
                       't10k-images-idx3-ubyte.gz',
                       't10k-labels-idx1-ubyte.gz']

DATA_PATH = Path.cwd().parent / "Neural Networks" / "01.MNIST Classification" / "00_MNIST_data"


class ParamSearch():
    def __init__(self, search_type: str):
        self.search_type = search_type
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.model = None
        self.params = None
        self.rand_search = None

    def fit(self):
        self._load_data()
        self._scale_data()

        # model
        self.model = KNeighborsClassifier()

        # define distribution of parameters
        self.params = [{'n_neighbors': [3, 10, 50, 100]},
                       # {'algorithm': ['auto', 'ball_tree', 'kd_tree']},
                       # {'leaf_size': [10, 30, 80, 200]}]
                       {'weights': ['distance', 'uniform']}]

        if self.search_type == 'grid':
            self._fit_grid_search()
        elif self.search_type == 'random':
            self._fit_randomized_search()
        else:
            raise ValueError(f'Search type entered as {self.search_type}. Select only `grid` or `random`.')

        self.rand_search.fit(self.train_images, self.train_labels)

        self._print_outs()

    def _load_data(self, data_path: Path = DATA_PATH):
        data = []
        for file in IMPORTED_DATA_FILES:
            with gzip.open(data_path / file, 'r') as f:
                data.append(idx2numpy.convert_from_file(f))

        self.train_images = data.pop(0)
        self.train_labels = data.pop(0)
        self.test_images = data.pop(0)
        self.test_labels = data.pop(0)

    def _scale_data(self, train_file_num: int = 10000, test_file_num: int = 1000):
        # reduce data
        (self.train_images, self.train_labels) = reduce_date(
            self.train_images, self.train_labels, train_file_num)
        (self.test_images, self.test_labels) = reduce_date(
            self.test_images, self.test_labels, test_file_num)

        # normalize data
        self.train_images = scale_data(self.train_images)
        self.test_images = scale_data(self.test_images)

        # reshape data
        self.train_images = np.reshape(self.train_images, (self.train_images.shape[0], -1))
        self.test_images = np.reshape(self.test_images, (self.test_images.shape[0], -1))

    def _fit_randomized_search(self):
        self.rand_search = RandomizedSearchCV(self.model, self.params,
                                              n_iter=3, cv=5,
                                              scoring='neg_mean_squared_error')

    def _fit_grid_search(self):
        self.rand_search = GridSearchCV(self.model, self.params,
                                        # n_iter=3,
                                        cv=3,
                                        scoring='neg_mean_squared_error',
                                        verbose=3,
                                        return_train_score=True)

    def _print_outs(self):
        # report the result of grid search
        print(f'Best params:\n{self.rand_search.best_params_}\n')
        print(f'Best Score:\n{self.rand_search.best_score_}\n')
        print(f'Best estimator:\n{self.rand_search.best_estimator_}\n')

        # report the scores of each estimator in the grid/rand search
        cv_result = self.rand_search.cv_results_
        pprint(cv_result.keys())

        # pick the best model
        final_model = self.rand_search.best_estimator_

        # final_prediction = final_model.predict(test_images)

        acc_final = final_model.score(self.test_images, self.test_labels)
        print(f'\nMean accuracy of final model = {round(acc_final, 4)}')
