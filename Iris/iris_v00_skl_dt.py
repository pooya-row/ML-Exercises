import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv('00_data/iris.data',
                   names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# split data
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1),
                                                    data['class'], test_size=0.2)

# build a tree model
model = DecisionTreeClassifier()

# train the model
model.fit(X_train, y_train)

# prediction on the test dataset
y_pred = model.predict(X_test)

# classification report
print(classification_report(y_test, y_pred))

# visualize
sns.pairplot(data, hue='class')
plt.show()
