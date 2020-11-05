import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('00_data/iris.data', names=['sepal length',
                                               'sepal width',
                                               'petal length',
                                               'petal width',
                                               'class'])
# data info
print(data.info())

# visualize
# sns.pairplot(data, hue='class')
# plt.show()

# split data
X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1),
                                                    data['class'], test_size=0.2)

print(X_test.info())
print(y_test)
