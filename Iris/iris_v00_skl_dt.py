import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree

data = pd.read_csv('00_data/iris.data',
                   names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

print(data.head())

# split data
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1),
                                                    data['class'], test_size=0.2)

# build a tree model
model = DecisionTreeClassifier(random_state=0, max_depth=2, splitter='best')#, min_samples_split=2)

# train the model
model.fit(X_train, y_train)

# prediction on the test dataset
y_pred = model.predict(X_test)

# classification report
print(classification_report(y_test, y_pred))

# visualize
# tree.plot_tree(model)
feature_name = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W']
class_name = ['Setosa', 'Versicolor', 'Virginica']

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
tree.plot_tree(model, feature_names=feature_name, class_names=class_name, filled=True)
plt.show()
