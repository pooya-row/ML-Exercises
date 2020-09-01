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

# # plot raw data
# plt.scatter(X, Y)
# plt.show()

# calculate a & b
a = (X.dot(Y) - len(X) * X.mean() * Y.mean()) / (X.dot(X) - X.mean() ** 2 * len(X))
b = Y.mean() - a * X.mean()

# calculate predictions
Y_hat = a * X + b

# calculate R^2
del1 = Y - Y.mean()
del2 = Y - Y_hat
R2 = 1 - del2.dot(del2) / del1.dot(del1)
# # alternative method
# TSS = sum((Y - Y.mean()) ** 2)
# RSS = sum((Y - Y_hat) ** 2)
# R2 = 1 - RSS / TSS

# plot data and fitted line
plt.scatter(X, Y)
plt.plot(X, Y_hat, c='red')
plt.show()

print(f'a = {a}')
print(f'b = {b}')
print(f'R² = {R2}')
