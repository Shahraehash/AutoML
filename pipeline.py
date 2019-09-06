# Dependencies
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from estimators import estimators
from feature_selection import featureSelectors
from hyperparameters import hyperParameterRange
from scalers import scalers

# Define the cross validator
cv = StratifiedKFold(n_splits=10)

class Debug(BaseEstimator, TransformerMixin):
    def transform(self, X):
        print('\tNumber of items: %d\n\tNumber of features: %d' % (X.shape[0], X.shape[1]))
        return X

    def fit(self, X, Y=None, **fit_params):
        return self

# Generate a pipeline
def generatePipeline(scaler, featureSelector, estimator, scoring='accuracy'):
    steps = []

    if scaler and scalers[scaler]:
        steps.append(('scaler', scalers[scaler]))

    if featureSelector and featureSelectors[featureSelector]:
        steps.append(('feature_selector', featureSelectors[featureSelector]))

    steps.append(('debug', Debug()))

    if estimator in hyperParameterRange:
        steps.append(('estimator', GridSearchCV(
                estimators[estimator],
                hyperParameterRange[estimator],
                return_train_score='False',
                cv=cv,
                n_jobs=-1,
                scoring=scoring
            )))
    else:
        steps.append(('estimator', estimators[estimator]))
    return Pipeline(steps)