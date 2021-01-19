import gzip
import idx2numpy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage.interpolation import shift


# normalizer function
def scale_data(x):
    return x / 255.


# reduce data function
def reduce_date(x, y, n=5120):
    ix = np.random.randint(0, x.shape[0], n)
    return x[ix], y[ix]


# import data
files = ['train-images-idx3-ubyte.gz',
         'train-labels-idx1-ubyte.gz',
         't10k-images-idx3-ubyte.gz',
         't10k-labels-idx1-ubyte.gz']

path = 'C:\\Users\\Pooya\\Documents\\Machine Learning\\ML-Exercise\\' \
       'Neural Networks\\01.MNIST Classification\\00_MNIST_data\\'

data = []
for file in files:
    with gzip.open(path + file, 'r') as f:
        data.append(idx2numpy.convert_from_file(f))

train_images = data.pop(0)
train_labels = data.pop(0)
test_images = data.pop(0)
test_labels = data.pop(0)

# reduce date
# (train_images, train_labels) = reduce_date(train_images, train_labels, 10000)
# (test_images, test_labels) = reduce_date(test_images, test_labels, 1000)

# normalize data
train_images = scale_data(train_images)
test_images = scale_data(test_images)

# data augmentation
directions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
train_images_aug = np.copy(train_images)
train_labels_aug = np.copy(train_labels)
for dr in directions:
    train_images_aug = \
        np.concatenate((train_images_aug,
                        np.array(list(map(lambda i: shift(i, dr, cval=0), train_images)))))

    train_labels_aug = np.concatenate((train_labels_aug, train_labels))

# shuffle images
shuffle_idx = np.random.permutation(len(train_images_aug))
train_images_aug = train_images_aug[shuffle_idx]
train_labels_aug = train_labels_aug[shuffle_idx]

# reshape data
train_images_aug = np.reshape(train_images_aug, (train_images_aug.shape[0], -1))
test_images = np.reshape(test_images, (test_images.shape[0], -1))

# best model from ex1
k_neighbors = KNeighborsClassifier(n_neighbors=4, weights='distance')
k_neighbors.fit(train_images_aug, train_labels_aug)

acc_final = k_neighbors.score(test_images, test_labels)
print(f'\nMean accuracy of final model = {round(acc_final, 4)}')  # 0.9763
