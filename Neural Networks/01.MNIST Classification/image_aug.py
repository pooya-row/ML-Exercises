import utils
import os
import numpy as np
import matplotlib.pyplot as plt

files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

path = os.getcwd() + '\\00_MNIST_data\\'

# load MNIST from local compressed files
train_images, train_labels, test_images, test_labels = utils.MNIST_import(path, files)

# scale input data
scaled_train_images, scaled_test_images = utils.scale_data(train_images, test_images)

# randomly chose an image from test set then plot image
random_inx = np.random.choice(scaled_test_images.shape[0])
test_image = scaled_test_images[random_inx]
plt.imshow(test_image, cmap='Greys')
plt.show()