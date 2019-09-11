# Dependencies
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA

from rffi import RandomForestFeatureImportanceSelector

featureSelectors = {
    'none': '',
    'pca-25': PCA(n_components=.25, svd_solver='full'),
    'pca-50': PCA(n_components=.50, svd_solver='full'),
    'pca-75': PCA(n_components=.75, svd_solver='full'),
    'rf-25': RandomForestFeatureImportanceSelector(n=.25),
    'rf-50': RandomForestFeatureImportanceSelector(n=.50),
    'rf-75': RandomForestFeatureImportanceSelector(n=.75),
    'select-25': SelectPercentile(percentile=25),
    'select-50': SelectPercentile(percentile=50),
    'select-75': SelectPercentile(percentile=75)
}

featureSelectorNames = {
    'none': 'all features (no feature selection)',
    'pca-25': 'principal component analysis (25%)',
    'pca-50': 'principal component analysis (50%)',
    'pca-75': 'principal component analysis (75%)',
    'rf-25': 'random forest importance (25%)',
    'rf-50': 'random forest importance (50%)',
    'rf-75': 'random forest importance (75%)',
    'select-25': 'select percentile (25%)',
    'select-50': 'select percentile (50%)',
    'select-75': 'select percentile (75%)'
}