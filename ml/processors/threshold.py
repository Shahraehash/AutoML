import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)

class Threshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold: float):
        self.estimator_ = model
        self.classes_ = self.estimator_.classes_
        self.threshold = threshold

    #pylint: disable = unused-argument
    def fit(self, x, y=None, **fit_params):
        """Do nothing"""

        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        predicate = self.estimator_.predict_proba(X)[:, 1] > self.threshold
        return np.where(predicate, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
