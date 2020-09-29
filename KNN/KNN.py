import numpy as np
import time
from util import get_data


class Knn:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        Y_hat = []
        for i, x_ts in enumerate(x):  # for test point
            dist = []  # k nearest distances
            for j, x_tn in enumerate(self.x):  # for training points
                diff = x_tn - x_ts  # calculate distance between training and test points
                sd = diff.dot(diff)  # calculate the sum of distance
                dist.sort()  # sort recorded k nearest distances
                if len(dist) < self.k:  # have we recorded k number of distances
                    dist.append((sd, self.y[j, 0]))
                elif dist[-1][0] > sd:  # if closer point is found record it
                    dist.pop()  # delete the farthest recorded distance
                    dist.append((sd, self.y[j, 0]))  # record the distance and its class

            clss = []
            for votes in dist:
                clss.append(votes[1])

            voted_class = max(tuple(clss), key=clss.count)
            Y_hat.append(voted_class)

        Y_hat = np.array(Y_hat)
        return Y_hat.reshape((len(Y_hat), 1))

    def score(self, lbl, pdt):
        # self.pdt = pdt
        # self.lbl = lbl
        return np.mean(lbl == pdt)


if __name__ == '__main__':
    X, Y = get_data(2000)  # pick 2000 rows of the main data
    Ntrain = 1000  # split the dataset into training and test datasets
    Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain, :]
    Xtest, Ytest = X[Ntrain + 850:, :], Y[Ntrain + 850:, :]

    accu = {}
    for k in (1, 2, 3, 4, 5):
        t0 = time.time()
        knn = Knn(k)
        knn.fit(Xtrain, Ytrain)
        P = knn.predict(Xtrain)
        accu[k] = (round(knn.score(Ytrain, P), 2), round(time.time()-t0, 3))

    print(accu)

    # knn = Knn(2)
    # knn.fit(Xtrain, Ytrain)
    # prid = knn.predict(Xtest)
    # P = knn.predict(Xtest)
    # pprint(Ytest)
    # pprint(P)
    # print(knn.score(Ytest, P))

# print(prid)
# print(type(prid))
# print(Ytest)
# print(type(Ytest))
