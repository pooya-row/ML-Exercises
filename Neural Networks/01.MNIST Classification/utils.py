import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import gzip
import idx2numpy


# custom callback class for prediction metrics
class PredictionCallback(Callback):
    """
    This object creates callbacks which generate the classification report
    as well as the confusion matrix for a given epoch and a set of (data, label).

    :param data: input images '(num, pixel, pixel, channel)'
    :param label: images known labels (num, class label)
    """

    def __init__(self, data, label):
        super().__init__()
        self.data = data  # input images
        self.label = label  # image labels

    def on_train_begin(self, logs=None):
        """
        initiate collectors at the beginning of training
        """
        self.confusion = []  # list collector for confusion matrices
        self.report = []  # list collector for classification reports
        self.last_epoch = 0

    # at the end of each epoch append classification report and
    # confusion matrix of that epoch to the collectors
    def on_epoch_end(self, epoch, logs=None):
        """
        At the end of each epoch append classification report and
        confusion matrix of that epoch to the collectors
        """

        y_hat = self.model.predict(self.data)

        self.report.append(
            classification_report(y_true=self.label,
                                  y_pred=np.argmax(y_hat, axis=1),
                                  digits=3))

        self.confusion.append(
            confusion_matrix(y_true=self.label,
                             y_pred=np.argmax(y_hat, axis=1)))

        self.last_epoch = epoch + 1


# data importer
def MNIST_import(path, files):
    """
    This function gets a list of compressed *.gz binary-like files and
     returns them in form of numpy arrays.

    :param path: absolute address of the input files
    :param files: list of 4 files in this order
     (train_images, train_labels, t10k_images, t10k_labels)
    :return: 4 numpy arrays in the same order of the files
    """
    data = []

    for file in files:
        with gzip.open(path + file, 'r') as f:
            data.append(idx2numpy.convert_from_file(f))

    train_images = data.pop(0)
    train_labels = data.pop(0)
    t10k_images = data.pop(0)
    t10k_labels = data.pop(0)

    return train_images, train_labels, t10k_images, t10k_labels


# scale the data
def scale_data(x_train, x_test):
    x_train = x_train / 255.
    x_test = x_test / 255.
    return x_train, x_test


# reduce data
def reduce_date(x, y, n=5120):
    """
    This function takes in two lists keeps only the first n elements of them
    :param x: list
    :param y: list
    :param n: int
    :return: truncated x-list, truncated y-list
    """
    return x[:n], y[:n]


# build the model
def get_model(num_layers, input_shape):
    """
    This function builds a Sequential model according to the below specification:
        1. A 2D convolutional layer with a 5x5 kernel and 4 filters; no padding; ReLU activation function
        2. MaxPooling with 3x3 kernel
        2. n 2D convolutional layers with 3x3 kernels and 8 filters; no padding; ReLU activation function
        3. MaxPooling layers, one after each of above Conv2Ds with 3x3 window, and strides of 1 (total of n)
        4. A flatten layer, which unrolls the input into a column tensor
        5. A dense output layer with 10 units and the softmax activation function
    :param input_shape: tuple (pixel, pixel, channel=1)
    :param num_layers: int, number of layers
    :return: Sequential model object
    """

    # instantiate the model
    model = tf.keras.models.Sequential()

    # add layers
    model.add(Conv2D(4, kernel_size=5, padding='valid',
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), strides=1))

    for i in range(num_layers):
        model.add(Conv2D(16, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=1))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model


# compile the model
def compile_model(mdl, metrics):
    """
    This function takes in the model returned from get_model function, and compiles it with
        * SGD optimiser (with default settings)
        * cross-entropy loss function
        * accuracy as the only metric.
    This function does not return anything; the model will be compiled in-place.

    :param mdl: Sequential model from `get_model`
    :param metrics: list, list of metrics to be monitored and recorded
    :return: None
    """

    mdl.compile(optimizer='SGD',
                loss='sparse_categorical_crossentropy',
                metrics=metrics)


# train the model
def train_model(mdl, images, labels, val_images, val_labels, num_e, batch_size, call_backs):
    """
    This function trains the model for `num_e` epochs on the `scaled_train_images` and train_labels
    and returns the training history, as returned by `model.fit`.

    :param mdl: Sequential model from `get_model`
    :param images: numpy.ndarray (n_train, pixel, pixel, channel=1)
    :param labels: numpy.ndarray (n_train, 1)
    :param val_images: validation data, numpy.ndarray (n_val, pixel, pixel, channel=1)
    :param val_labels: validation labels, numpy.ndarray (n_val, 1)
    :param num_e: int number of epochs
    :param batch_size: int, size of mini batches
    :param call_backs: list of callbacks for model saving and early-stopping
    :return: model.fit.history
    """
    if call_backs == []:
        history = mdl.fit(images, labels,
                          validation_data=(val_images, val_labels),
                          epochs=num_e, batch_size=batch_size)
    else:
        history = mdl.fit(images, labels,
                          validation_data=(val_images, val_labels),
                          epochs=num_e, batch_size=batch_size,
                          callbacks=call_backs)
    return history


# evaluate the model
def evaluate_model(mdl, images, labels):
    """
    This function evaluates the model on the `scaled_test_images` and `test_labels`.
    and returns a tuple (test_loss, test_accuracy).

    :param mdl: Sequential model from 'get_model'
    :param images: numpy.ndarray (n_test, pixel, pixel, channel=1)
    :param labels: numpy.ndarray (n_test,1)
    :return: tuple (test_loss, test_accuracy)
    """
    test_loss, test_accuracy = mdl.evaluate(images, labels)
    return test_loss, test_accuracy


# image cropping
def remove_image_margin(image):
    """
    This function removes the blank margin of a given single-channel image
    :param image: input image as a numpy array (ndarray)
    :return: output image with no margins as a numpy array (ndarray)
    """

    # from skimage.transform import rescale, resize, downscale_local_mean
    # image_rescaled = rescale(cropped_img, .5, anti_aliasing=False)
    # image_resized = resize(cropped_img, (cropped_img.shape[0] // .5, cropped_img.shape[1] // .5),
    #                        anti_aliasing=True)
    # image_downscaled = downscale_local_mean(cropped_img, (2, 3))

    cropped_img = np.copy(image)

    # remove left blank
    empty_left_col = 0
    while image[:, empty_left_col].max() == 0:
        cropped_img = np.delete(cropped_img, 0, 1)
        empty_left_col += 1

    # remove right blank
    empty_right_col = len(image[1]) - 1
    while image[:, empty_right_col].max() == 0:
        cropped_img = np.delete(cropped_img, -1, 1)
        empty_right_col -= 1

    # remove top blank
    empty_top_row = 0
    while image[empty_top_row, :].max() == 0:
        cropped_img = np.delete(cropped_img, 0, 0)
        empty_top_row += 1

    # remove bottom blank
    empty_bottom_row = len(image[0]) - 1
    while image[empty_bottom_row, :].max() == 0:
        cropped_img = np.delete(cropped_img, -1, 0)
        empty_bottom_row -= 1

    return cropped_img
