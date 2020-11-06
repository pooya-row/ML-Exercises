import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import utils
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from datetime import datetime

files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

path = os.getcwd() + '\\00_MNIST_data\\'

# load MNIST data directly from keras
# mnist_data = tf.keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()
# load MNIST from local compressed files
train_images, train_labels, test_images, test_labels = utils.MNIST_import(path, files)

# # reduce training date
# (train_images, train_labels) = utils.reduce_date(train_images, train_labels, 5120)
# # reduce testing date
# (test_images, test_labels) = utils.reduce_date(test_images, test_labels, 300)

# scale input data
scaled_train_images, scaled_test_images = utils.scale_data(train_images, test_images)

# add a dummy channel dimension
scaled_train_images = scaled_train_images[..., np.newaxis]
scaled_test_images = scaled_test_images[..., np.newaxis]

# split training set to create validation set
scaled_train_images, scaled_val_images, train_labels, val_labels = \
    train_test_split(scaled_train_images, train_labels, test_size=0.15)

# initialize the plot
fig1, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
fig1.subplots_adjust(bottom=0.2, wspace=0.3)

# initiate the metric recorder for plot
metric_plot = {'layers': [], 'test_acc': [], 'train_t': [], 'test_t': []}

# model parameters
num_epoch = 25
batch_size = 128
min_num_layers = 2
max_num_layers = 4
# early stopping settings
early_stop = EarlyStopping(monitor='accuracy', patience=2, min_delta=0.005)
# initiate the list of number of epochs to record all loops n_of_e
num_of_epochs = []

# metrics to monitor and record in history
MET = ['accuracy']

# validation precision and recall collectors for plots
all_prec = []
all_rcll = []
max_epoch = []

for n in range(min_num_layers, max_num_layers + 1):
    # create the model
    model = utils.get_model(n, scaled_train_images[0].shape)

    # compile the model
    utils.compile_model(model, MET)

    # create path for saving model
    path = f'01.CNN - Saved Model/{n}-Layers/'
    # set model save settings
    checkpoint = ModelCheckpoint(path + 'CNN.Ep{epoch:02d}',
                                 save_weights_only=False, save_freq='epoch')
    # form callback list
    call_backs = [
        checkpoint,
        early_stop,
        utils.PredictionCallback(scaled_val_images, val_labels)
    ]

    # train the model with the scaled training images
    t0 = time.time()
    run_log = utils.train_model(model,
                                scaled_train_images, train_labels,
                                scaled_val_images, val_labels,
                                num_epoch, batch_size,
                                call_backs)
    t_train = round(time.time() - t0, 3)
    # retrieve history data
    history_data = pd.DataFrame(run_log.history)
    # print(history_data.head())

    # generate and format classification report
    max_epoch.append(call_backs[2].last_epoch)
    precision = []
    recall = []

    for i in range(call_backs[2].last_epoch):  # loop over all epochs for current  model
        report = " ".join(call_backs[2].report[i].split())  # remove extra spaces from report str

        cls_rep = []
        for num in report.split(' '):  # convert the str to a list
            cls_rep.append(num)

        cls_rep.insert(0, 'class')  # add title 'class' to the first row
        cls_rep = np.array(cls_rep[:55])  # 11row x 5col
        cls_rep = cls_rep.reshape((11, 5))
        precision.append(list(map(float, cls_rep[1:, 1])))  # separate precision column
        recall.append(list(map(float, cls_rep[1:, 2])))  # separate recall column

    all_prec.extend(precision)  # keep epoch-wise precisions for current model
    all_rcll.extend(recall)  # keep epoch-wise recalls for current model

    # evaluate the model on scaled test images
    t0 = time.time()
    test_loss, test_accuracy = utils.evaluate_model(model,
                                                    scaled_test_images,
                                                    test_labels)
    t_test = round(time.time() - t0, 3)

    # print out loss and accuracy
    print(f'N = {n}')
    print(f'\tTest loss\t\t{test_loss:.3f}')
    print(f'\tTest accuracy\t{test_accuracy:.3f}')
    print(f'\tTrain time\t\t{t_train} s')
    print(f'\tTest time\t\t{t_test} s')
    print(f'\tLast Epoch #\t{call_backs[2].last_epoch}')
    print('\n')

    # record metrics
    metric_plot['layers'].append(n)
    metric_plot['test_acc'].append(round(test_accuracy, 4))
    metric_plot['train_t'].append(t_train)
    metric_plot['test_t'].append(t_test)

    # record number of epochs for this loop
    num_of_epochs.append(len(history_data['val_accuracy']))
    # plot validation accuracy vs epochs
    axes[1].plot(range(1, num_of_epochs[-1] + 1), 'val_accuracy',
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
axes[1].set_ylabel('Validation Accuracy')
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

# plot and format precision vs classes for each epoch and each model config
fig2, axes2 = plt.subplots(nrows=1,
                           ncols=(max_num_layers - min_num_layers) + 1,
                           figsize=(14, 4))
fig2.subplots_adjust(bottom=0.2, wspace=0.3)

col = -1
for i in range((max_num_layers - min_num_layers) + 1):
    line = []
    for j in range(max_epoch[i]):
        col += 1
        ll, = axes2[i].plot(range(10), all_prec[col], marker='o', label='Epoch #' + str(j+1))
        line.append(ll)
        axes2[i].legend(line, [l.get_label() for l in line], loc='best')

    axes2[i].set_title(f'Model with {min_num_layers + i} hidden layers')
    axes2[i].set_xlabel('Classes')
    axes2[i].set_ylabel('Validation Precision')
    axes2[i].tick_params(axis='both', bottom=False, left=False)  # remove ticks
    axes2[i].grid(True)
    axes2[i].set_xlim([0, 9])
    axes2[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

# plot and format recall vs classes for each epoch and each model config
fig3, axes3 = plt.subplots(nrows=1,
                           ncols=(max_num_layers - min_num_layers) + 1,
                           figsize=(14, 4))
fig3.subplots_adjust(bottom=0.2, wspace=0.3)

col = -1
for i in range((max_num_layers - min_num_layers) + 1):
    line = []
    for j in range(max_epoch[i]):
        col += 1
        ll, = axes3[i].plot(range(10), all_rcll[col], marker='o', label='Epoch #' + str(j+1))
        line.append(ll)
        axes3[i].legend(line, [l.get_label() for l in line], loc='best')

    axes3[i].set_title(f'Model with {min_num_layers + i} hidden layers')
    axes3[i].set_xlabel('Classes')
    axes3[i].set_ylabel('Validation Recall')
    axes3[i].tick_params(axis='both', bottom=False, left=False)  # remove ticks
    axes3[i].grid(True)
    axes3[i].set_xlim([0, 9])
    axes3[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

# save all plots
now = datetime.now()
t = now.strftime('%d-%m-%y %H.%M.%S')
fig1.savefig(f'01.CNN - Saved Model/01.CNN {t}.png', dpi=300)
fig2.savefig(f'01.CNN - Saved Model/Precision {t}.png', dpi=300)
fig3.savefig(f'01.CNN - Saved Model/Recall {t}.png', dpi=300)
# print('\n', len(all_prec))
# print('\n', all_prec)
# print('\n', len(all_rcll))
# print('\n', all_rcll)
# plt.show()
