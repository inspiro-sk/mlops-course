import os
import yaml
import sys


import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, 'mappings/config.yaml')

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = os.path.join(ROOT_DIR, config['DATASET_PATH'])
data = pd.read_csv(DATA_PATH)

# define target variable
target = data[config['TARGET']]

# drop unneccessary variables
data.drop(config['FEATURES_TO_DROP'], axis=1, inplace=True)

# use only selected features
selected_features = config['NUMERIC_FEATURES'] + config['CATEGORICAL_FEATURES']
data = data[selected_features]

# split train and test set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=config['SEED'])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train[config['CATEGORICAL_VARS_WITH_NA_MISSING']].isnull().sum() > 0)
print(X_train[config['CATEGORICAL_VARS_WITH_NA_FREQUENT']].isnull().sum() > 0)
print(X_train[config['NUMERIC_FEATURES']].isnull().sum() > 0)

pipeline = Pipeline([
    # IMPUTATION
    ('missing_imputation', CategoricalImputer(
        imputation_method='missing', variables=config['CATEGORICAL_VARS_WITH_NA_MISSING']
    )),
    ('frequent_imputation', CategoricalImputer(
        imputation_method='frequent', variables=config['CATEGORICAL_VARS_WITH_NA_FREQUENT']
    )),
    ('mean_imputation', MeanMedianImputer(
        imputation_method='mean', variables=config['NUMERIC_FEATURES']
    ))

    # TEMPORAL VARIABLES
])

pipeline.fit(X_train)

X_train= pipeline.transform(X_train)
X_test = pipeline.transform(X_test)


print(X_train[config['CATEGORICAL_VARS_WITH_NA_MISSING']].isnull().sum() > 0)
print(X_train[config['CATEGORICAL_VARS_WITH_NA_FREQUENT']].isnull().sum() > 0)
print(X_train[config['NUMERIC_FEATURES']].isnull().sum() > 0)
