import numpy as np
import matplotlib.pyplot as plt

# generate data
N = 50
X = np.linspace(2, 20, N)
Y = 0.5 * X + np.random.randn(N)
Y[-1] += 20
Y[-2] += 20

# X = X.reshape(N, 1)
# X = np.concatenate((X, np.ones((N, 1))), axis=1)
X = np.vstack((X, np.ones(N))).T
w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_hat = X.dot(w)

# calculate R-squared using its definition in simple lin. reg.
d1 = (Y - Y.mean()).dot(Y - Y.mean())
d2 = (Y - Y_hat).dot(Y - Y_hat)
print(f'Without regularization R² = {round(1 - d2 / d1, 5)}')

# plot data
plt.scatter(X[:, 0], Y, label='data', c='black')
plt.plot(X[:, 0], Y_hat, label='No Regularization', ls='--')

# iterate for different values of lambda
for l2 in range(0, 5100, 500):
    # don't consider l2 = 0
    if l2 == 0:
        continue

    # L2 regularized
    w_l2 = np.linalg.solve(l2 * np.eye(2) + X.T.dot(X), X.T.dot(Y))
    Y_hat_l2 = X.dot(w_l2)

    # calculate R-squared for each value of lambda
    d2 = (Y - Y_hat_l2).dot(Y - Y_hat_l2)
    print(f'For λ = {l2},\tR² = {round(1 - d2 / d1, 5)}')

    # plot data
    plt.plot(X[:, 0], Y_hat_l2, label=f'L2 = {l2}')

plt.legend()
plt.show()
