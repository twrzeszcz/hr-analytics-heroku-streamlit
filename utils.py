from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p



class custom_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        X_new = X.copy()
        self.cols = X_new.isna().sum()[X_new.isna().sum() != 0].index

        return self

    def transform(self, X):
        X_new = X.copy()
        X_new.drop('enrollee_id', axis=1, inplace=True)
        for col in self.cols:
            X_new.loc[X_new[col].isna(), col] = 'Not specified'

        return X_new


class skewness_remover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        X_new = X.copy()
        self.skewness = X_new[X_new.describe().columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        self.skewness = self.skewness[self.skewness > 0.75]

        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.skewness.index:
            X_new[col] = boxcox1p(X_new[col], 0.0)

        return X_new