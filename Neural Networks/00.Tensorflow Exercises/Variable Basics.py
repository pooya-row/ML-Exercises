import tensorflow as tf
import numpy as np

# define one
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)
print(my_variable, '\n')

# just like a tensor
print("Shape:\t", my_variable.shape)
print("DType:\t", my_variable.dtype)
print("As NumPy:\n", my_variable.numpy(), '\n')
# print("A variable:", my_variable)
print("my_tensor:", my_tensor, '\n')
print("Viewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("\n", tf.reshape(my_variable, ([1, 4])), '\n')

# re-assigning to a variable
a = tf.Variable([2.0, 3.0])
a.assign([1, 2])  # it keeps the same dtype 'a' had when initialized
print(a.numpy())
print(a.assign_add([2, 3]).numpy())  # ops and assign
print(a.assign_sub([9, 4]).numpy())  # ops and assign
print(a.numpy(), '\n')

# variables can have names. Two variables can have same name
aa = tf.Variable(my_tensor, name="Mark")
bb = tf.Variable(my_tensor + 1, name="Mark")
# These are element-wise unequal, despite having the same name
print(aa == bb)

# gradients
with tf.GradientTape() as tape:
    y = aa ** 3
dy_da = tape.gradient(y, aa)
print('\n', aa.numpy(), '\n\n', dy_da.numpy(), '\n')

# typical differentiation
w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]

with tf.GradientTape(persistent=True) as tape:
    y = tf.matmul(x, w) + b
    loss = tf.reduce_mean(y ** 2)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])
print(w.shape, '\n', dl_dw.shape, '\n')
print('dl/dw = ', dl_dw.numpy(), '\n')

# using a dictionary to pass the variables
my_vars = {
    'w': tf.Variable(tf.random.normal((3, 2)), name='w'),
    'b': tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')}
# x = [[1., 2., 3.]]

with tf.GradientTape(persistent=True) as tape:
    y = tf.matmul(x, my_vars['w']) + my_vars['b']
    loss = tf.reduce_mean(y ** 2)

grad = tape.gradient(loss, my_vars)
print('dl/db = ', grad['b'].numpy(), '\n')

# A trainable variable
x0 = tf.Variable(3.0, name='x0')
# Not trainable
x1 = tf.Variable(3.0, name='x1', trainable=False)
# Not a Variable: A variable + tensor returns a tensor.
x2 = tf.Variable(4., name='pooya')
# x2 = x2.assign_add(1)
# Not a variable
x3 = tf.constant(3.0, name='x3')

with tf.GradientTape() as tape1:
    tape1.watch([x1,x2, x3])
    y = (x0 ** 2) + (x1 ** 3) + (x2 ** 4) + (x3 ** 5)
grad1 = tape1.gradient(y, [x0, x1, x2, x3])
print([var.name for var in tape1.watched_variables()])
for g in grad1:
    print(g)

# control flow
x = tf.constant([1.0, -1.])
v0 = tf.Variable([2., 5.])
v1 = tf.Variable([4., 2.0])
result=[]
with tf.GradientTape(persistent=True) as tape2:
    tape2.watch(x)
    for i in x:

        if i < 0.0:
            rr = v0
        else:
            rr = v1 ** 2
        result.append(rr)
dv0, dv1 = tape2.gradient(result, [v0, v1])
print('\n',result)
print(dv0)
print(dv1)

