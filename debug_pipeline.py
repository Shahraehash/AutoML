from sklearn.base import TransformerMixin, BaseEstimator

class Debug(BaseEstimator, TransformerMixin):
    def transform(self, X):
        print('\tNumber of items: %d\n\tNumber of features: %d' % (X.shape[0], X.shape[1]))
        return X

    def fit(self, X, Y=None, **fit_params):
        return self
