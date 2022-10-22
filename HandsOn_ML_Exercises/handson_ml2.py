import numpy as np
import pandas as pd
import tarfile
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from scipy import stats


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_hh = X[:, room_ix] / X[:, hh_ix]
        ppl_per_hh = X[:, pop_ix] / X[:, hh_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_hh, ppl_per_hh, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_hh, ppl_per_hh]


# get the data
data_path = 'dataset/'

# extract
with tarfile.open(data_path + 'housing.tgz') as f:
    
    import os
    
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(f, data_path)

# read
with open(data_path + 'housing.csv', 'r') as f:
    housing = pd.read_csv(f)

# stratified split based on median household income
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0, 1.5, 3, 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_ix, test_ix in sss.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_ix]
    strat_test_set = housing.loc[test_ix]

# separate labels
tr_housing = strat_train_set.drop('median_house_value', axis=1)
tr_label = strat_train_set['median_house_value'].copy()

# choose relevant columns
col_names = 'total_rooms', 'total_bedrooms', 'population', 'households'
room_ix, bedroom_ix, pop_ix, hh_ix = [
    tr_housing.columns.get_loc(c) for c in col_names]

# transformation pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # fill in missing data
    ('feature_adder', CombinedAttributesAdder()),  # add engineered features
    ('std_scaler', StandardScaler())  # standardize
])

# transformation pipeline for both numerical nad categorical features
num_features = list(tr_housing)  # list of numerical feature names
num_features.pop(-2)
cat_features = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', OneHotEncoder(), cat_features)
])

housing_prepared = full_pipeline.fit_transform(tr_housing)  # numpy array

# linear regression model
lin_reg = LinearRegression()

# train
lin_reg.fit(housing_prepared, tr_label)

# prepare some data to run prediction on
some_data = tr_housing.iloc[np.random.randint(0, housing_prepared.shape[0], 5)]
some_label = tr_label.iloc[np.random.randint(0, housing_prepared.shape[0], 5)]
some_data_prep = full_pipeline.transform(some_data)

# make predictions
# print('Prediction:\t', list(lin_reg.predict(some_data_prep)))
# print('Labels:\t\t', list(some_label))

# compute error
housing_prediction = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(tr_label, housing_prediction)
lin_rmse = np.sqrt(lin_mse)
print(f'Linear regression RMSE = {lin_rmse}')

# decision tree regression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, tr_label)

tree_prediction = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(tr_label, tree_prediction)
tree_rmse = np.sqrt(tree_mse)
print(f'Decision tree regression RMSE = {tree_rmse}')


# function to calculate and display evaluation a model the dataset
def display_scores(model, X, y):
    scores = cross_val_score(model, cv=10,
                             X=X, y=y,
                             scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print('Scores:\n', rmse_scores)
    print('Mean:\t', round(rmse_scores.mean(), 2))
    print('Std dev:', round(rmse_scores.std(), 2))
    return round(rmse_scores.mean(), 2), round(rmse_scores.std(), 2)


print('\nLinear regression model:')
display_scores(lin_reg, housing_prepared, tr_label)
print('\nDecision tree model:')
display_scores(tree_reg, housing_prepared, tr_label)

# random forest model
forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, tr_label)
# print('\nRandom forest regression model:')
# display_scores(forest_reg, housing_prepared, tr_label)

# fine-tuning via grid search
forest_params = [
    {'n_estimators': [5, 40], 'max_features': [6, 10]},
    {'bootstrap': [False], 'n_estimators': [2], 'max_features': [8]}
]

grid_search = GridSearchCV(forest_reg, forest_params,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, tr_label)

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)

cv_result = grid_search.cv_results_
print(cv_result.keys())
for sc, par in zip(cv_result['mean_test_score'], cv_result['params']):
    print(round(np.sqrt(-sc), 1), par)

feature_importance = grid_search.best_estimator_.feature_importances_

extra_features = ['room_per_hh', 'pop_per_hh', 'bedroom_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_1hot = list(cat_encoder.categories_[0])
features = num_features + extra_features + cat_1hot
pprint(sorted(zip(feature_importance, features), reverse=True))

final_model = grid_search.best_estimator_

# evaluate on test set
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_prediction = final_model.predict(X_test_prepared)

# final evaluation score
final_mse = mean_squared_error(y_test, final_prediction)
final_rmse = np.sqrt(final_mse)
print(f'\nEvaluation score on test set: {round(final_rmse, 1)}')

# confidence interval
conf = 0.95
sqd_er = (final_prediction - y_test) ** 2
conf_int = np.sqrt(stats.t.interval(conf, len(sqd_er) - 1,
                                    loc=sqd_er.mean(),
                                    scale=stats.sem(sqd_er)))
print(f'\n95% Confidence interval = {conf_int}')
