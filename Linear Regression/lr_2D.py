import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

X = []
Y = []
for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

# plot data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# calculate R-squared using Cor(Y, Y_hat)^2
nom = (Y - Y.mean()).dot((Y_hat - Y_hat.mean()))
den1 = (Y - Y.mean()).dot((Y - Y.mean()))
den2 = (Y_hat - Y_hat.mean()).dot((Y_hat - Y_hat.mean()))
R2 = nom ** 2 / (den1 * den2)
print(f'R² = {R2} based on R²=Cor(Y, Y_hat)².')

# calculate R-squared using its definition in simple lin. reg.
d1 = (Y - Y.mean()).dot(Y - Y.mean())
d2 = (Y - Y_hat).dot(Y - Y_hat)
print(f'R² = {1 - d2 / d1} based on its definition in simple lin. reg. (1-RSS/TSS).')
