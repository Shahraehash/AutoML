from estimators import estimatorNames
from feature_selection import featureSelectorNames
from scalers import scalerNames
from scorers import scorerNames

def modelKeyToName(key):
    scaler, featureSelector, estimator, scorer = key.split('__')
    return estimatorNames[estimator] + ' model using ' + scorerNames[scorer] + ' scored grid search with ' + scalerNames[scaler] + ' and with ' + featureSelectorNames[featureSelector]