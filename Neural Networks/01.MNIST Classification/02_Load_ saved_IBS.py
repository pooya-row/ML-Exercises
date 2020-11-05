import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils import scale_data
from pprint import pprint

# load MNIST data directly from keras
mnist_data = tf.keras.datasets.mnist
_, (test_images, test_labels) = mnist_data.load_data()

# scale input data
_, scaled_test_images = scale_data(test_images, test_images)

# add a dummy channel dimension
scaled_test_images = scaled_test_images[..., np.newaxis]

# randomly chose an image from test set - plot image
random_inx = np.random.choice(scaled_test_images.shape[0])
test_image = scaled_test_images[random_inx]
plt.imshow(test_image, cmap='Greys')

# load the model
n = 2  # model with this number of layers
ep = 5  # epoch number
model = load_model(f'02_IBS-Saved Model/{n}-Layers/IBS_Ep{ep:02d}')

# use the model to predict the label of the chosen image
prediction = model.predict(test_image[np.newaxis, ...])

print(f"Random index is:\t{random_inx}")  # print random index
print(f"Actual Label is:\t{test_labels[random_inx]}")  # print label
print(f'Model prediction:\t{np.argmax(prediction)}')  # print predicted label
pprint(prediction)
plt.show()
