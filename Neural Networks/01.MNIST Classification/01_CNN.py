import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from datetime import datetime
import utils

# load MNIST data directly from keras
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# reduce training date
(train_images, train_labels) = utils.reduce_date(train_images, train_labels, 5000)
# reduce testing date
(test_images, test_labels) = utils.reduce_date(test_images, test_labels, 500)

# scale input data
scaled_train_images, scaled_test_images = utils.scale_data(train_images, test_images)

# add a dummy channel dimension
scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

# initialize the plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
plt.subplots_adjust(bottom=0.2, wspace=0.3)

# initiate the metric record for plot
metric_plot = {'layers': [], 'test_acc': [], 'train_t': [], 'test_t': []}

# model parameters
num_epoch = 50
batch_size = 64
min_num_layers = 2
max_num_layers = 4
# early stopping settings
early_stop = EarlyStopping(monitor='accuracy', patience=2, min_delta=0.005)
# initiate the list of number of epochs to record all loops n_of_e
num_of_epochs = []

for n in range(min_num_layers, max_num_layers + 1):
    # create the model
    model = utils.get_model(n, scaled_train_images[0].shape)

    # compile the model
    utils.compile_model(model)

    # create path for saving model
    path = f'01.CNN - Saved Model/{n}-Layers/'
    # set model save settings
    checkpoint = ModelCheckpoint(path + 'CNN.Ep{epoch:02d}',
                                 save_weights_only=False, save_freq='epoch')
    # form callback list
    call_backs = [checkpoint, early_stop]

    # train the model with the scaled training images
    t0 = time.time()
    run_log = utils.train_model(model, scaled_train_images, train_labels,
                                num_epoch, batch_size, call_backs)
    t_train = round(time.time() - t0, 3)
    history_data = pd.DataFrame(run_log.history)

    # evaluate the model on scaled test images
    t0 = time.time()
    test_loss, test_accuracy = utils.evaluate_model(model, scaled_test_images,
                                                    test_labels)
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
