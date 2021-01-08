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
from sklearn.svm import SVR
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


# get the data
data_path = 'dataset/'

# extract
with tarfile.open(data_path + 'housing.tgz') as f:
    f.extractall(data_path)

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

# transformation pipeline for both numerical and categorical features
num_features = list(tr_housing)  # list of numerical feature names
num_features.pop(-2)
cat_features = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', OneHotEncoder(), cat_features)
])

housing_prepared = full_pipeline.fit_transform(tr_housing)  # numpy array

# linear regression model
sv_reg = SVR()

# display_scores(sv_reg, housing_prepared, tr_label)

# fine-tuning the model (grid search)
svr_params = [
    {'kernel': ['linear'], 'C': [1000., 250.]},
    # {'kernel': ['linear', 'rbf'], 'C': [10., 250.], 'gamma': ['auto', 'scale']}
]
# define grid search
grid_search = GridSearchCV(sv_reg, svr_params,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
# run grid search
grid_search.fit(housing_prepared, tr_label)

# report the result of grid search
print(grid_search.best_params_)
print(f'\n{grid_search.best_score_}')
print(f'\n{grid_search.best_estimator_}\n')

# report the scores of each estimator in the grid search
cv_result = grid_search.cv_results_
pprint(cv_result.keys())

for sc, par, mft, sft, mst, sst in zip(cv_result['mean_test_score'],
                                       cv_result['params'],
                                       cv_result['mean_fit_time'],
                                       cv_result['std_fit_time'],
                                       cv_result['mean_score_time'],
                                       cv_result['std_score_time']):
    print('\nmean_test_score:\t', round(np.sqrt(-sc), 1),
          '\nparams:\t', par,
          '\nmean_fit_time:\t', mft, '\nstd_fit_time:\t', sft,
          '\nmean_score_time:\t', mst, '\nstd_score_time:\t', sst)

# report the importance of each dataset feature when used by the best estimator
# feature_importance = grid_search.best_estimator_.feature_importances_
# extra_features = ['room_per_hh', 'pop_per_hh', 'bedroom_per_room']
# cat_encoder = full_pipeline.named_transformers_['cat']
# cat_1hot = list(cat_encoder.categories_[0])
# features = num_features + extra_features + cat_1hot
# pprint(sorted(zip(feature_importance, features), reverse=True))

# pick the best model
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
