import pandas as pd
import data_plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# function to assign proper age average to passengers with missing age data
def fill_in_age(df):
    # calculate age mean for each passenger class
    age_means = []
    for i in range(1, 4):
        age_means.append(
            round(train_data[train_data['Pclass'] == i]['Age'].mean(), 1))

    if pd.isnull(df['Age']):
        if df['Pclass'] == 1:
            return age_means[0]
        elif df['Pclass'] == 2:
            return age_means[1]
        elif df['Pclass'] == 3:
            return age_means[2]
    return df['Age']


# load data
train_data = pd.read_csv('00_data/train.csv')
test_data = pd.read_csv('00_data/test.csv')

# plot for data exploration
# plot how much data is missing
# data_plot.missing_data(train_data)
#
# explore data by plotting
# data_plot.data_explore(train_data)

# fill in the missing age data based on average age of passenger class
train_data['Age'] = train_data.apply(fill_in_age, axis=1)
sex = pd.get_dummies(train_data['Sex'], drop_first=True)
embark = pd.get_dummies(train_data['Embarked'], drop_first=True)

train_data = pd.concat([
    train_data[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']],
    sex, embark], axis=1)

# split test dataset
X_train, X_val, y_train, y_val = train_test_split(
    train_data.drop(['Survived', 'PassengerId'], axis=1),
    train_data['Survived'],
    test_size=0.2
)

# print(X_train)
# print(X_val)

# build a logistic regression model
model = LogisticRegression()

# train the model
model.fit(X_train, y_train)

# predictions based on validation dataset
predictions = model.predict(X_val)

# classification report
print(classification_report(y_val, predictions))
