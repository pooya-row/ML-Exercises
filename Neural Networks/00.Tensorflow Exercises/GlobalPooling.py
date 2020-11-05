import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from datetime import datetime
import utils
from tensorflow import keras
from sklearn.model_selection import train_test_split


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
    model.add(Conv2D(4, kernel_size=5, padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), strides=1))
    # model.add(GlobalAveragePooling2D())

    for i in range(num_layers):
        model.add(Conv2D(9, kernel_size=3, padding='valid', activation='relu'))
        model.add(MaxPooling2D((3, 3), strides=1))

    model.add(GlobalAveragePooling2D(name='Pooya'))
    # model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model


# load MNIST data directly from keras
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# reduce training date
train_images, train_labels = utils.reduce_date(train_images, train_labels, 10240)
# reduce testing date
test_images, test_labels = utils.reduce_date(test_images, test_labels, 500)

# scale input data
scaled_train_images, scaled_test_images = utils.scale_data(train_images, test_images)

# add a dummy channel dimension
scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

# create validation set
scaled_train_images, scaled_val_images, train_labels, val_labels = \
    train_test_split(scaled_train_images, train_labels, test_size=0.15)

# initialize the plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
plt.subplots_adjust(bottom=.2, wspace=.3)

# initiate the metric record for plot
metric_plot = {'layers': [], 'test_acc': [], 'train_t': [], 'test_t': []}

# model parameters
num_epoch = 3
batch_size = 64
min_num_layers = 2
max_num_layers = 3
# early stopping settings
early_stop = EarlyStopping(monitor='accuracy', patience=2, min_delta=0.005)
# initiate the list of number of epochs to record all loops n_of_e
num_of_epochs = []

print(scaled_train_images[0].shape)

for n in range(min_num_layers, max_num_layers + 1):
    # create the model
    model = get_model(n, scaled_train_images[0].shape)

    layer_name = 'Pooya'
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(scaled_test_images)

    # compile the model
    utils.compile_model(model, ['accuracy']) #, 'val_accuracy'

    # create path for saving model
    path = f'GlobalPooling/{n}-Layers/'
    # set model save settings
    checkpoint = ModelCheckpoint(path + 'GP.Ep{epoch:02d}',
                                 save_weights_only=False, save_freq='epoch')
    # form callback list
    call_backs = [checkpoint, early_stop]

    # train the model with the scaled training images
    t0 = time.time()
    run_log = utils.train_model(model, scaled_train_images, train_labels,
                                scaled_val_images, val_labels,
                                num_epoch, batch_size, call_backs)
    t_train = round(time.time() - t0, 3)
    history_data = pd.DataFrame(run_log.history)
    print(history_data)

    # evaluate the model on scaled test images
    t0 = time.time()
    test_loss, test_accuracy = utils.evaluate_model(model, scaled_test_images, test_labels)
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
    num_of_epochs.append(len(history_data['val_accuracy']))
    # plot accuracy vs epochs
    axes[1].plot(range(1, num_of_epochs[-1] + 1), 'val_accuracy',
                 data=history_data, marker='o', label=str(n) + '-layer')

    # print(intermediate_output)
    print(type(intermediate_output))
    print(intermediate_output.shape)
    hh=(intermediate_output.numpy())
    print(type(hh))

# plot test accuracy vs # of layers
axes[0].plot('layers', 'test_acc', data=metric_plot, marker='o')
axes[0].set_xlabel('# of Layers')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_xlim([min_num_layers - 1, max_num_layers + 1])
axes[0].set_ylim([.9, 1.])

# plot training accuracy vs # of epochs for models with different number of layers
axes[1].legend(loc='best')
axes[1].set_xlabel('Epoch Number')
axes[1].set_ylabel('Train VAL Accuracy')
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
plt.savefig(f'GlobalPooling/GP {t}.png', dpi=300)

plt.show()
