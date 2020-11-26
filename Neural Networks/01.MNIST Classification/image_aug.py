import utils
import os
import numpy as np
import matplotlib.pyplot as plt

# input files
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

# crop image
cropped_img = utils.remove_image_margin(test_image)

# print shape of the cropped image
print(f'The cropped image size is: {cropped_img.shape}')

# plot both original and cropped images
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(test_image, cmap='Greys')
axes[1].imshow(cropped_img, cmap='Greys')
axes[0].get_xaxis().set_ticks([])
axes[0].get_yaxis().set_ticks([])
axes[1].get_xaxis().set_ticks([])
axes[1].get_yaxis().set_ticks([])
plt.show()
