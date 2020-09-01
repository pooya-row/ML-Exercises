import numpy as np
import matplotlib.pyplot as plt

# import data into lists
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# convert data into array
X = np.array(X)
Y = np.array(Y)

# # plot array data
# plt.scatter(X, Y)
# plt.show()

# calculate a & b
a = (X.dot(Y) - len(X) * X.mean() * Y.mean()) / (X.dot(X) - X.mean() ** 2 * len(X))
b = Y.mean() - a * X.mean()
# print(f'a = {a}')
# print(f'b = {b}')

# calculate predictions
Y_hat = a * X + b

# plot data and fitted line
plt.scatter(X, Y)
plt.plot(X, Y_hat, c='red')
plt.show()
