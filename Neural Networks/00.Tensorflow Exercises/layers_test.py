import tensorflow as tf
# import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
# from util import *
import sys
print(sys.path)


# mnist_data = tf.keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()
#
# # scale input data
# scaled_train_images, scaled_test_images = scale_data(train_images, test_images)
#
# # add a dummy channel dimension for Conv2D
# scaled_train_images = scaled_train_images[..., np.newaxis]
# scaled_test_images = scaled_test_images[..., np.newaxis]

model = tf.keras.models.Sequential()
# add layers
model.add(Conv2D(4, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))

for i in range(2):
    model.add(Conv2D((i+2)*8, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))

# model.add(GlobalAveragePooling2D())
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

print(model.compute_output_shape((250, 28, 28, 1)))
model.add_metric()