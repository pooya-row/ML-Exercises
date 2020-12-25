import numpy as np
import matplotlib.pyplot as plt
import utils
import os
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# load MNIST from local compressed files
files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz']

path = os.getcwd() + '\\00_MNIST_data\\'

train_images, train_labels, test_images, test_labels = utils.MNIST_import(path, files)
print(type(test_images))
# reduce date
train_images, train_labels = utils.reduce_date(train_images, train_labels, 5120)
test_images, test_labels = utils.reduce_date(test_images, test_labels, 300)

# scale input data
scaled_train_images, scaled_test_images = utils.scale_data(train_images, test_images)

# split training set to create validation set
scaled_train_images, scaled_val_images, train_labels, val_labels = \
    train_test_split(scaled_train_images, train_labels, test_size=0.15)

# image augmentation
# blurred = np.zeros(scaled_train_images.shape)  # Gaussian blur
# cropped = np.zeros(scaled_train_images.shape)  # crop, square, rescale
aug_images = np.zeros(tuple((scaled_train_images.shape[0] * 2,)) +
                      scaled_train_images.shape[1:])

print(type(aug_images))

for img in range(scaled_train_images.shape[0]):
    # blurred[img] = gaussian(scaled_train_images[img], sigma=1.15)
    # cropped[img] = utils.crop_rescale(scaled_train_images[img])
    aug_images[2 * img - 1] = gaussian(scaled_train_images[img], sigma=1.15)
    aug_images[2 * img] = utils.crop_rescale(scaled_train_images[img])

print(aug_images.shape)
# plot
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
# fig.subplots_adjust(hspace=.6, wspace=.4)
# ind = 1253
# axes[0].imshow(scaled_train_images[ind], cmap='Greys')
# axes[1].imshow(blurred[ind], cmap='Greys')
# axes[2].imshow(cropped[ind], cmap='Greys')
# plt.show()

# add a dummy dimension for color channel
# blurred = blurred[..., np.newaxis]
# cropped = cropped[..., np.newaxis]
aug_images = aug_images[..., np.newaxis]

import tensorflow.keras.backend as kb


def NT_Xent(y_actual, y_pred):
    custom_loss = kb.square(y_actual - y_pred)
    return custom_loss


# define encoder and head models
# rep_model = tf.keras.models.Sequential()
# rep_model.add(Flatten(input_shape=aug_images[0].shape))
# for n in range(3):
#     rep_model.add(Dense(20, activation='relu'))
#
# print(rep_model.summary())
#
# # define head model
# head_model = tf.keras.models.Sequential()
# head_model.add(Dense(12, activation='relu', input_shape=(20,)))
# head_model.add(Dense(10, activation='softmax'))
#
# print(head_model.summary())

# # compile the encoder and ead models
# rep_model.compile(optimizer='SGD',
#                   loss='',
#                   metrics='acc')
#
# head_model.compile(optimizer='SGD',
#                    loss='sparse_categorical_crossentropy',
#                    metrics='acc')
#
# # train the encoder
# num_e = 100
# batch_size = 4096
# history = rep_model.fit(cropped,  # labels,
#                         epochs=num_e, batch_size=batch_size,
#                         # validation_data=(val_images, val_labels),
#                         # callbacks=call_backs
#                         )
