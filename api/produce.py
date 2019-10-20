"""
Creates the model based on the pipeline components (provided by
the key, hyper parameters and features used).
"""

from sklearn.pipeline import Pipeline

from .processors.estimators import ESTIMATORS
from .processors.feature_selection import FEATURE_SELECTORS
from .processors.scalers import SCALERS
from .import_data import import_train
from .utils import explode_key

def create_model(key, hyper_parameters, features, train_set=None, label_column=None):
    """Refits the requested model and pickles it for export"""

    if train_set is None:
        print('Missing training data')
        return {}

    if label_column is None:
        print('Missing column name for classifier target')
        return {}

    # Import data
    (x_train, _, y_train, _, _) = import_train(train_set, label_column)

    # Get pipeline details from the key
    scaler, feature_selector, estimator, _, _ = explode_key(key)

    # Create the pipeline
    steps = []

    if scaler and SCALERS[scaler]:
        steps.append(('scaler', SCALERS[scaler].fit(x_train, y_train)))

    if 'pca-' in feature_selector:
        steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))
#    elif feature_selector != 'none':
        

    steps.append(('estimator', ESTIMATORS[estimator].set_params(hyper_parameters)))

    return Pipeline(steps)
