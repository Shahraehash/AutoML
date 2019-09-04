# Dependencies
from sklearn.pipeline import Pipeline

from estimators import estimators
from feature_selection import featureSelectors
from scalers import scalers

# Generate a pipeline
def generatePipeline(scaler, featureSelector, estimator):
    steps = []

    if scaler and scalers[scaler]:
        steps.append((scaler, scalers[scaler]))

    if featureSelector and featureSelectors[featureSelector]:
        steps.append((featureSelector, featureSelectors[featureSelector]))

    steps.append((estimator, estimators[estimator]))
    return Pipeline(steps)