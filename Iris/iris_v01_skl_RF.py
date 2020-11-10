import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pprint import pprint
from sklearn import tree

# import data
data = pd.read_csv('00_data/iris.data',
                   names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# split data
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1),
                                                    data['class'], test_size=0.2)

# get model
model = RandomForestClassifier(n_estimators=20)

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# classification report
print(classification_report(y_test, y_pred))

# list details of each estimator tree
# pprint(model.estimators_)

# visualize a set of estimator trees
feature_name = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W']
class_name = ['Setosa', 'Versicolor', 'Virginica']

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 5))

for index in range(0, 5):
    tree.plot_tree(model.estimators_[index],
                   feature_names=feature_name,
                   class_names=class_name,
                   filled=True,
                   ax=axes[index])
    axes[index].set_title('Estimator: ' + str(index), fontsize=11)

plt.show()
