import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tolerance=0.05, variables=None) -> None:
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.tolerance = tolerance
        self.variables = variables

    def fit(self, X, y=None):
        self.encoder_dict_ = {}

        for feature in self.variables:
            # encoder will learn the most frequent categories
            t = pd.Series(X[feature].value_counts(normalize=True))

            # frequent labels:
            self.encoder_dict_[feature] = list(t[t >= self.tolerance].index)
        
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                    X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"] # here we add target into dataframe columns

        self.encoder_dict_ = {}

        for feature in self.variables:
            # group dataframe by feature, calculate mean of target and sort smallest to largest
            t = temp.groupby([feature])["target"].mean().sort_values(ascending=True).index

            # create dictionary where each value is assigned number in sequence starting from 0
            # the higher the target mean for particular value the higher the number
            # this is ordinal encoding
            self.encoder_dict_[feature] = {k: i for i, k in enumerate(t, 0)}


    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X

