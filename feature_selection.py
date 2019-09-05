# Dependencies
from sklearn.feature_selection import SelectPercentile

featureSelectors = {
    'none': '',
    'select-50': SelectPercentile(percentile=50),
    'select-75': SelectPercentile(percentile=75)
}

featureSelectorNames = {
    'none': 'all features (no feature selection)',
    'select-50': 'select percentile (50%)',
    'select-75': 'select percentile (75%)'
}