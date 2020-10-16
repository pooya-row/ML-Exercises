import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from datetime import datetime


# scale the data
def scale_data(x_train, x_test):
    x_train = x_train / 255.
    x_test = x_test / 255.
    return x_train, x_test


# reduce data
def reduce_date(x, y, n=5000):
    '''
    This function takes in two lists keeps only the first n elements of them
    :param x: list
    :param y: list
    :param n: int
    :return: list, list
    '''
    return x[:n], y[:n]


# build the model
def get_model(num_layers, input_shape):
    '''
    This function builds a Sequential model according to the below specification:
        1. A 2D convolutional layer with a 5x5 kernel and 8 filters; zero padding; ReLU activation function
        2. n 2D convolutional layers with 3x3 kernels and 8 filters; zero padding; ReLU activation function
        3. n+1 maxpooling layers, with 3x3 window, and strides of 1 after each of the layers above
        4. A flatten layer, which unrolls the input into a one-dimensional tensor
        5. A dense output layer with 10 units and the softmax activation function
    :param input_shape: tuple (pixel, pixel, channel=1)
    :param num_layers: int, number of layers
    :return: the model
    '''

    # instantiate the model
    model = tf.keras.models.Sequential()

    # add layers
    model.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), strides=1))

    for i in range(num_layers):
        model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=1))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model


# compile the model
def compile_model(mdl):
    '''
    This function takes in the model returned from get_model function, and compiles it with
        * SGD optimiser (with default settings)
        * cross-entropy loss function
        * accuracy as the only metric.
    This function does not return anything; the model will be compiled in-place.
    :param mdl: Sequential model from 'get_model'
    :return: None
    '''
    mdl.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# train the model
def train_model(mdl, images, labels, num_e, batch_size, call_backs):
    '''
    This function trains the model for num_e epochs on the scaled_train_images and train_labels
    and returns the training history, as returned by model.fit.

    :param mdl: Sequential model from 'get_model'
    :param images: numpy.ndarray (n_train, pixel, pixel, channel=1)
    :param labels: numpy.ndarray (n_train,1)
    :param num_e: int number of epochs
    :param batch_size: int, size of mini batches
    :param call_backs: list of callbacks for model saving and early-stopping
    :return: model.fit.history
    '''
    if call_backs == []:
        history = mdl.fit(images, labels, epochs=num_e, batch_size=batch_size)
    else:
        history = mdl.fit(images, labels, epochs=num_e,
                          batch_size=batch_size, callbacks=call_backs)
    return history


# evaluate the model
def evaluate_model(mdl, images, labels):
    '''
    This function evaluates the model on the scaled_test_images and test_labels.
    and returns a tuple (test_loss, test_accuracy).

    :param mdl: Sequential model from 'get_model'
    :param images: numpy.ndarray (n_test, pixel, pixel, channel=1)
    :param labels: numpy.ndarray (n_test,1)
    :return: tuple (test_loss, test_accuracy)
    '''
    test_loss, test_accuracy = mdl.evaluate(images, labels)
    return test_loss, test_accuracy


# load MNIST data directly from keras
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# # reduce training date
# (train_images, train_labels) = reduce_date(train_images, train_labels, 2000)
# # reduce testing date
# (test_images, test_labels) = reduce_date(test_images, test_labels, 100)

# scale input data
scaled_train_images, scaled_test_images = scale_data(train_images, test_images)

# add a dummy channel dimension
scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

# initialize the plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
plt.subplots_adjust(bottom=.2, wspace=.3)

# initiate the metric record for plot
metric_plot = {'layers': [], 'test_acc': [], 'train_t': [], 'test_t': []}

# model parameters
num_epoch = 50
batch_size = 128
min_num_layers = 2
max_num_layers = 5
# early stopping settings
early_stop = EarlyStopping(monitor='accuracy', patience=1, min_delta=0.005)
# initiate the list of number of epochs to record all loops n_of_e
num_of_epochs = []

for n in range(min_num_layers, max_num_layers + 1):
    # create the model
    model = get_model(n, scaled_train_images[0].shape)

    # compile the model
    compile_model(model)

    # create path for saving model
    path = f'01.CNN - Saved Model/{n}-Layers/'
    # set model save settings
    checkpoint = ModelCheckpoint(path + 'CNN.Ep{epoch:02d}',
                                 save_weights_only=False, save_freq='epoch')
    # form callback list
    call_backs = [checkpoint, early_stop]

    # train the model with the scaled training images
    t0 = time.time()
    run_log = train_model(model, scaled_train_images, train_labels,
                          num_epoch, batch_size, call_backs)
    t_train = round(time.time() - t0, 3)
    history_data = pd.DataFrame(run_log.history)

    # evaluate the model on scaled test images
    t0 = time.time()
    test_loss, test_accuracy = evaluate_model(model, scaled_test_images, test_labels)
    t_test = round(time.time() - t0, 3)

    # print out loss and accuracy
    print(f'N = {n}')
    print(f'\tTest loss =\t{test_loss:.3f}')
    print(f'\tTest accuracy =\t{test_accuracy:.3f}')
    print(f'\tTrain time = {t_train}')
    print(f'\tTest time = {t_test}')
    print('\n')

    # record metrics
    metric_plot['layers'].append(n)
    metric_plot['test_acc'].append(round(test_accuracy, 4))
    metric_plot['train_t'].append(t_train)
    metric_plot['test_t'].append(t_test)

    # record number of epochs for this loop
    num_of_epochs.append(len(history_data['accuracy']))
    # plot accuracy vs epochs
    axes[1].plot(range(1, num_of_epochs[-1] + 1), 'accuracy',
                 data=history_data, marker='o', label=str(n) + '-layer')

# plot test accuracy vs # of layers
axes[0].plot('layers', 'test_acc', data=metric_plot, marker='o')
axes[0].set_xlabel('# of Layers')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_xlim([min_num_layers - 1, max_num_layers + 1])
axes[0].set_ylim([.9, 1.])

# plot training accuracy vs # of epochs for models with different number of layers
axes[1].legend(loc='best')
axes[1].set_xlabel('Epoch Number')
axes[1].set_ylabel('Train Accuracy')
axes[1].set_xlim([0, max(num_of_epochs) + 1])
axes[1].set_ylim([.8, 1.])
axes[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

# plot training and test time vs # of layers
ax2 = axes[2].twinx()  # set a secondary axis
axes[2].set_xlim([min_num_layers - 1, max_num_layers + 1])
axes[2].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
axes[2].set_xlabel('# of Layers')
axes[2].set_ylabel('Train CPU Time (s)')
ax2.set_ylabel('Test CPU Time (s)')
ax2.tick_params(axis='both', right=False)
l1, = axes[2].plot('layers', 'train_t', data=metric_plot, marker='o', label='Train')
l2, = ax2.plot('layers', 'test_t', data=metric_plot, marker='o', c='r', label='Test')
lines = [l1, l2]
axes[2].legend(lines, [l.get_label() for l in lines], loc='best')

# format the plot
for i in range(3):
    axes[i].tick_params(axis='both', bottom=False, left=False)  # remove ticks
    axes[i].grid(True)  # add grids

# save the plot
now = datetime.now()
t = now.strftime('%d-%m-%y %H.%M.%S')
plt.savefig(f'01.CNN - Saved Model/01.CNN {t}.png', dpi=300)

# plt.show()
