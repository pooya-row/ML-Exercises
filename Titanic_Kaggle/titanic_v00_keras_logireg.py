import pandas as pd
import data_plot
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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


# step function
def step_func(x):
    if x < 0.5:
        return 0
    return 1


# learning rate decay scheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.2)


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
    train_data.drop('Survived', axis=1),
    train_data['Survived'],
    test_size=0.3
)

# build a logistic regression model
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(8,), kernel_regularizer='l2')
    # Dense(8, activation='relu'),
    # Dense(8, activation='relu'),
    # Dense(8, activation='relu'),
    # Dense(1, activation='sigmoid')
])

# compile the model
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# define callback for learning rate decay based on epoch number
call_back = tf.keras.callbacks.LearningRateScheduler(scheduler)

# train the model
history = model.fit(
    X_train.drop('PassengerId', axis=1),
    y_train,
    epochs=10,
    batch_size=8,
    callbacks=[call_back],
    verbose=0
)

history_data = pd.DataFrame(history.history)

# evaluate the model on X_val
test_loss, test_accuracy = model.evaluate(
    X_val.drop('PassengerId', axis=1),
    y_val,
    verbose=0
)

# calculate classification report based on validation dataset
predictions = model.predict(X_val.drop('PassengerId', axis=1))
# discretize predictions
predictions = list(map(step_func, predictions))
cls_rep = classification_report(y_val, predictions)

# printouts
print(history_data.head(3), '\n')
print(history_data.tail(3), '\n')
print(f'Evaluation results:\n\tTest loss\t\t{test_loss:.3f}')
print(f'\tTest accuracy\t{test_accuracy:.3f}')
print('\n', cls_rep)
