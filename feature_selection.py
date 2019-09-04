# Dependencies
from sklearn.feature_selection import SelectPercentile

featureSelectors = {
    'none': '',
    'select-75': SelectPercentile(percentile=75)
}

featureSelectorNames = {
    'none': 'no feature selection',
    'select-75': 'select percentile (75%)'
}