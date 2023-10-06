import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# Create a transformer that calculates difference between
# reference variable and list of variables passed
# e.g. between YrSold and other 'year' variables in our case
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.reference_variable = reference_variable

    # This transform does not learn anything from the data
    # therefore fit just returns self
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X
