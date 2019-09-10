# Dependencies
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA

featureSelectors = {
    'none': '',
    'pca-50': PCA(n_components=.50, svd_solver='full'),
    'pca-75': PCA(n_components=.75, svd_solver='full'),
    'select-50': SelectPercentile(percentile=50),
    'select-75': SelectPercentile(percentile=75)
}

featureSelectorNames = {
    'none': 'all features (no feature selection)',
    'pca-50': 'principal component analysis (50%)',
    'pca-75': 'principal component analysis (75%)',
    'select-50': 'select percentile (50%)',
    'select-75': 'select percentile (75%)'
}