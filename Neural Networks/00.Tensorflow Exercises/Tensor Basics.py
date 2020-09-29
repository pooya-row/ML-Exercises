import numpy as np
import tensorflow as tf

# rank-0 tensor
rank_0 = tf.constant(3)
# print(rank_0)

# rank-1 tensor
rank_1 = tf.constant([3.4, 5, 8])
# print(rank_1)

# rank-2 tensor
rank_2 = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
# print(rank_2)

# rank-3 tensor
rank_3 = tf.constant([
    [[3, 8, 5, 8],
     [1, 5, 0, 0],
     [-4, 3, 44, 0]],
    [[4, 75, 6, 3],
     [-4, 3, 4, 0],
     [1, 5, 0, 0]]
])
# print(rank_3, '\n')

# convert to numpy array
print(rank_2.numpy())
print(type(np.array(rank_2)), '\n')
print(rank_3[1, 0, 1].numpy(), '\n')

# basic math operations
a = tf.ones([2, 2])
b = tf.constant([[3, 2], [6, 12.]])  # dtype of a & b have to match (int and float don't add up)
print(tf.add(a, b), '\n')  # or (a + b) element-wise
print(tf.multiply(a, b), '\n')  # or (a * b) element-wise
print(tf.matmul(a, b), '\n')  # or (a @ b) element-wise

# other operations
print(tf.reduce_max(rank_3), '\n')
print(tf.argmax(rank_3[0]), '\n')
print(tf.nn.softmax(rank_2), '\n')

rank_4_tensor = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy(), '\n')

# tensor shape
print(rank_3.shape)
print(rank_3.shape.as_list())

# tensor reshape
reshaped = tf.reshape(rank_3, [3, 8])
print(reshaped)
print(rank_3)
print(tf.reshape(reshaped, [2, -1]))

# change data type
ff = tf.constant([2.8, -7.9, 3.2, -4.1])
print(ff)
print(tf.cast(ff, dtype=tf.float64), '\n')

# broadcasting
x = tf.constant([[1], [2], [3]])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y), '\n')
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]), '\n')

# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))
