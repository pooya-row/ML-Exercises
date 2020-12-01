import utils
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import difference_of_gaussians, unsharp_mask, gaussian
from skimage.transform import rescale, resize, downscale_local_mean
import gzip
import idx2numpy
import os

# input files and scale input data
with gzip.open(os.getcwd() + '\\00_MNIST_data\\t10k-images-idx3-ubyte.gz', 'r') as f:
    test_images = idx2numpy.convert_from_file(f)
with gzip.open(os.getcwd() + '\\00_MNIST_data\\t10k-labels-idx1-ubyte.gz', 'r') as f:
    test_labels = idx2numpy.convert_from_file(f)

scaled_test_images = test_images / 255.

# randomly chose an image from test set then plot image
random_inx = np.random.choice(scaled_test_images.shape[0])
# random_inx = 8700 # 9828
test_image = scaled_test_images[random_inx]

# crop image
cropped_img = utils.remove_image_margins(test_image)

# apply a Gaussian blur filter
filtered_cr_img = gaussian(test_image, sigma=1.1)

# make the cropped image square
squared_cr_img = utils.make_square(cropped_img)

# rescale to match the original image
rescaled_sq_cr_img = rescale(squared_cr_img,
                             test_image.shape[0] / squared_cr_img.shape[0],
                             anti_aliasing=True)

# printouts
print(f'Cropped image size:\t{cropped_img.shape}')
print(f'Random index:\t\t{random_inx}')
print(f'True label is:\t\t{test_labels[random_inx]}')


# plot resulting images and histogram
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
fig.subplots_adjust(hspace=.6, wspace=.4)


# gs = axes[0, 0].get_gridspec()
# for ax in axes[:, 0]:  # remove the underlying axes
#     ax.remove()
# axbig = fig.add_subplot(gs[:, 0])  # merge the first column subplots

def image_vector(image):
    return list(image.reshape(image.shape[0] * image.shape[1]))


axes[0, 0].imshow(test_image, cmap='Blues')
axes[0, 1].hist(image_vector(test_image), bins=10)
axes[1, 0].imshow(cropped_img, cmap='Greys')
axes[1, 1].imshow(squared_cr_img, cmap='Greys')
axes[1, 2].imshow(rescaled_sq_cr_img, cmap='Reds')
axes[0, 2].imshow(filtered_cr_img, cmap='Reds')
axes[0, 0].set_title('Original')
axes[0, 1].set_title('Histogram')
axes[1, 0].set_title('Cropped (int.)')
axes[1, 1].set_title('Cropped & Squared (int.)')
axes[1, 2].set_title('Cropped & Squared \n& Resized (final)')
axes[0, 2].set_title('Blurred (final)')
plt.show()

# 2x3 plot
# plot resulting images
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
# fig.subplots_adjust(hspace=.5, wspace=.5)
# axes[0, 0].imshow(cropped_img, cmap='Greys')
# axes[1, 0].imshow(filtered_cr_img, cmap='Greys')
# axes[0, 1].imshow(squared_cr_img, cmap='Greys')
# axes[1, 1].imshow(squared_flt_cr_img, cmap='Greys')
# axes[0, 2].imshow(rescaled_sq_cr_img, cmap='Greys')
# axes[1, 2].imshow(rescaled_sq_flt_cr_img, cmap='Greys')
# axes[0, 0].set_title('Cropped')
# axes[1, 0].set_title('Cropped & Blurred')
# axes[0, 1].set_title('Cropped & Squared')
# axes[1, 1].set_title('Cropped & Blurred & Squared')
# axes[0, 2].set_title('Cropped & Squared & Resized')
# axes[1, 2].set_title('Cropped & Blurred & Squared & Resized')
# for ax in axes:
#     for i in range(3):
#         ax[i].get_xaxis().set_ticks([])
#         ax[i].get_yaxis().set_ticks([])
#
# plt.show()

# other filters
# from skimage.transform import rescale, resize, downscale_local_mean
# image_rescaled = rescale(cropped_img, .5, anti_aliasing=False)
# image_resized = resize(cropped_img, (cropped_img.shape[0] // .5, cropped_img.shape[1] // .5),
#                        anti_aliasing=True)
# image_downscaled = downscale_local_mean(cropped_img, (2, 3))
# filtered_image =unsharp_mask(test_image, radius=10, amount=0.5, preserve_range=True)
# filtered_image=gaussian(test_image, sigma=1.75)
# filtered_cr_img = difference_of_gaussians(cropped_img, low_sigma=1, high_sigma=2)
