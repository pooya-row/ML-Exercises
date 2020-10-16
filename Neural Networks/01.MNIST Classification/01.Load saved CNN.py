import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# scale the data
def scale_data(x_train, x_test):
    x_train = x_train / 255.
    x_test = x_test / 255.
    return x_train, x_test


# load MNIST data directly from keras
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# scale input data
scaled_train_images, scaled_test_images = scale_data(train_images, test_images)

# add a dummy channel dimension
scaled_test_images = scaled_test_images[..., np.newaxis]

# randomly chose an image from test set - plot image
random_inx = np.random.choice(scaled_test_images.shape[0])
test_image = scaled_test_images[random_inx]
plt.imshow(test_image, cmap='gray')

# load the model
n = 4  # model with this number of layers
ep = 5  # epoch number
model = load_model(f'01.CNN - Saved Model/{n}-Layers/CNN.Ep0{ep}')

# use the model to predict the label of the chosen image
prediction = model.predict(test_image[np.newaxis, ...])
print(f"Label is: {test_labels[random_inx]}")  # print label
print(prediction)
print(f'Model prediction: {np.argmax(prediction)}')  # print predicted label

plt.show()
