import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


# scale the data
def scale_mnist_data(train_images, test_images):
    train_images = train_images / 255
    test_images = test_images / 255
    return train_images, test_images


# reduce data
def reduce_date(x, y, n=6000):
    '''
    This function takes in two lists keeps only the first n elements of them
    :param x: list
    :param y: list
    :param n: int
    :return: list, list
    '''
    return x[:n], y[:n]


# build the model
def get_model(input_shape):
    '''
    This function builds a Sequential model according to the below specification:
        1. A 2D convolutional layer with a 3x3 kernel and 8 filters; zero padding; ReLU activation function.
        2. A max pooling layer, with a 2x2 window, and default strides.
        3. A flatten layer, which unrolls the input into a one-dimensional tensor.
        4. Two dense hidden layers, each with 64 units and ReLU activation functions.
        5. A dense output layer with 10 units and the softmax activation function.
    :param input_shape: tuple (pixel, pixel, channel=1)
    :return: the model
    '''

    # instantiate the model
    model = tf.keras.models.Sequential()

    # add layers
    model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model


# compile the model
def compile_model(model):
    '''
    This function takes in the model returned from get_model function, and compiles it with
        * adam optimiser (with default settings)
        * cross-entropy loss function
        * accuracy as the only metric.
    This function does not return anything; the model will be compiled in-place.
    :param model: Sequential model from 'get_model'
    :return: None
    '''
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# train the model
def train_model(model, scaled_train_images, train_labels):
    '''
    This function trains the model for 5 epochs on the scaled_train_images and train_labels
    and returns the training history, as returned by model.fit.

    :param model: Sequential model from 'get_model'
    :param scaled_train_images: numpy.ndarray (n_train, pixel, pixel, channel=1)
    :param train_labels: numpy.ndarray (n_train,1)
    :return: model.fit.history
    '''
    history = model.fit(scaled_train_images, train_labels, epochs=5)
    return history


# evaluate the model
def evaluate_model(model, scaled_test_images, test_labels):
    '''
    This function evaluates the model on the scaled_test_images and test_labels.
    and returns a tuple (test_loss, test_accuracy).

    :param model: Sequential model from 'get_model'
    :param scaled_test_images: numpy.ndarray (n_test, pixel, pixel, channel=1)
    :param test_labels: numpy.ndarray (n_test,1)
    :return: tuple (test_loss, test_accuracy)
    '''
    test_loss, test_accuracy = model.evaluate(scaled_test_images, test_labels)
    return test_loss, test_accuracy


# load MNIST data directly from keras
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# reduce the date
(train_images, train_labels) = reduce_date(train_images, train_labels, 2000)
(test_images, test_labels) = reduce_date(test_images, test_labels, 500)

# scale input data
scaled_train_images, scaled_test_images = scale_mnist_data(train_images, test_images)

# add a dummy channel dimension
scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

# create the model
model = get_model(scaled_train_images[0].shape)

# compile the model
compile_model(model)

# train the model with the scaled training images
run_log = train_model(model, scaled_train_images, train_labels)
history_data = pd.DataFrame(run_log.history)

# evaluate the model on scaled test images
test_loss, test_accuracy = evaluate_model(model, scaled_test_images, test_labels)

# print out loss and accuracy
print(f'Test loss =\t{test_loss:.3f}')
print(f'Test accuracy =\t{test_accuracy:.3f}')

# plot loss and accuracy vs epochs
ax = history_data.plot(secondary_y='loss', mark_right=False, marker='o')
ax.set_ylabel('Accuracy')
ax.right_ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')

# format x-axis labels
locator = matplotlib.ticker.MultipleLocator(1)
plt.gca().xaxis.set_major_locator(locator)
plt.show()
