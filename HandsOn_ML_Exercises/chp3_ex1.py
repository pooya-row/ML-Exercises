from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import idx2numpy
from pprint import pprint
import gzip

from HandsOn_ML_Exercises.utils import scale_data, reduce_date

IMPORTED_DATA_FILES = ['train-images-idx3-ubyte.gz',
                       'train-labels-idx1-ubyte.gz',
                       't10k-images-idx3-ubyte.gz',
                       't10k-labels-idx1-ubyte.gz']

DATA_PATH = Path.cwd().parent  / "Neural Networks" / "01.MNIST Classification" / "00_MNIST_data"

data = []
for file in IMPORTED_DATA_FILES:
    with gzip.open(DATA_PATH / file, 'r') as f:
        data.append(idx2numpy.convert_from_file(f))

train_images = data.pop(0)
train_labels = data.pop(0)
test_images = data.pop(0)
test_labels = data.pop(0)

(train_images, train_labels) = reduce_date(train_images, train_labels, 10000)
(test_images, test_labels) = reduce_date(test_images, test_labels, 1000)

# normalize data
train_images = scale_data(train_images)
test_images = scale_data(test_images)

# reshape data
train_images = np.reshape(train_images, (train_images.shape[0], -1))
test_images = np.reshape(test_images, (test_images.shape[0], -1))

# model
k_neighbors = KNeighborsClassifier()

# define distribution of parameters
params = [{'n_neighbors': [3, 10, 50, 100]},
          # {'algorithm': ['auto', 'ball_tree', 'kd_tree']},
          # {'leaf_size': [10, 30, 80, 200]}]
          {'weights': ['distance', 'uniform']}]

# rand_search = RandomizedSearchCV(k_neighbors, params,
#                                  n_iter=3, cv=5,
#                                  scoring='neg_mean_squared_error')
rand_search = GridSearchCV(k_neighbors, params,
                           # n_iter=3,
                           cv=3,
                           scoring='neg_mean_squared_error',
                           verbose=3,
                           return_train_score=True)

rand_search.fit(train_images, train_labels)

# report the result of grid search
print(f'Best params:\n{rand_search.best_params_}\n')
print(f'Best Score:\n{rand_search.best_score_}\n')
print(f'Best estimator:\n{rand_search.best_estimator_}\n')

# report the scores of each estimator in the grid/rand search
cv_result = rand_search.cv_results_
pprint(cv_result.keys())

# pick the best model
final_model = rand_search.best_estimator_

# final_prediction = final_model.predict(test_images)

acc_final = final_model.score(test_images, test_labels)
print(f'\nMean accuracy of final model = {round(acc_final, 4)}')
