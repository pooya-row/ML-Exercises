import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x * x])
    Y.append(float(y))

# convert to numpy array
X = np.array(X)
Y = np.array(Y)

# # plot the data
# plt.scatter(X[:, 1], Y)
# plt.show()

# solve for w in Xt.Y_hat = (Xt.X).w
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

# plot the predictions
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat), c='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
