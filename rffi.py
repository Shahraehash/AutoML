import math
from sklearn.base import TransformerMixin, BaseEstimator

from estimators import estimators

class RandomForestFeatureImportanceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n=.8):
        self.model = estimators['rf']
        self.n = n
    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self
    def transform(self, X):
        self.total = math.floor(self.n * X.shape[1])
        return X[:,self.get_top_features()]
    def get_top_features(self):
        return self.model.feature_importances_.argsort()[::-1][:self.total]