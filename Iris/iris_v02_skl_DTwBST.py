import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier

sns.set()

# import data
data = pd.read_csv('00_data/iris.data',
                   names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# split data
X_train, X_test, y_train, y_test = \
    train_test_split(data.drop('class', axis=1),
                     data['class'], test_size=0.2)

# build a tree estimator
model = DecisionTreeClassifier(random_state=0, max_depth=2)  # , splitter='best')#, min_samples_split=2)

n_estim = 20  # number of estimators for boosting
# build a boosting based on the tree estimator
clf = AdaBoostClassifier(base_estimator=model, n_estimators=n_estim)

# train the boosted tree
clf.fit(X_train, y_train)

# prediction on the test dataset using boosted tree
y_pred = clf.predict(X_test)

# classification metrics of boosted tree
print(f'Metrics of the boosted tree:\n{classification_report(y_test, y_pred)}\n')
# print(clf.estimator_errors_)
print(f'The estimators are weighted as follow:\n\t{clf.estimator_weights_}\n\n')

# a single tree for comparison
model.fit(X_train, y_train)  # train
y_pred_dt = model.predict(X_test)  # predict
print(f'Metrics of a single tree:\n{classification_report(y_test, y_pred_dt)}\n')
print(f'Test score of a single tree: {model.score(X_test, y_test)}')

# plot boosted tree error
plt.plot(clf.estimator_errors_)
plt.xlabel('Estimator Number')
plt.ylabel('Estimator Error')
plt.xlim((0, n_estim))
plt.show()
