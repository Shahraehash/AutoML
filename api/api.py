"""
Auto ML

Supervised learning using an exhaustive search of ideal pre-processing (if any), algorithms,
and hyper-parameters with feature engineering.
"""

# Dependencies
import os
import itertools

from dotenv import load_dotenv

from .processors.estimators import ESTIMATOR_NAMES
from .processors.feature_selection import FEATURE_SELECTOR_NAMES
from .processors.scalers import SCALER_NAMES
from .processors.searchers import SEARCHER_NAMES
from .processors.scorers import SCORER_NAMES
from .generalization import generalize
from .model import generate_model
from .import_data import import_data
from .pipeline import generate_pipeline
from .summary import print_summary
from .utils import model_key_to_name

# Load environment variables
load_dotenv()
IGNORE_ESTIMATOR = os.getenv('IGNORE_ESTIMATOR', '').split(',')
IGNORE_FEATURE_SELECTOR = os.getenv('IGNORE_FEATURE_SELECTOR', '').split(',')
IGNORE_SCALER = os.getenv('IGNORE_SCALER', '').split(',')
IGNORE_SEARCHERS = os.getenv('IGNORE_SEARCHERS', '').split(',')
IGNORE_SCORER = os.getenv('IGNORE_SCORER', '').split(',')

def find_best_model(train_set=None, test_set=None, labels=None, label_column=None):
    """Generates all possible models and outputs the generalization results"""

    if train_set is None:
        print('Missing training data')
        return {}

    if test_set is None:
        print('Missing test data')
        return {}

    if label_column is None:
        print('Missing column name for classifier target')
        return {}

    # Import data
    (x_train, y_train, x2, y2, feature_names) = import_data(train_set, test_set, label_column)

    results = {}
    models = 0

    all_pipelines = list(itertools.product(
        *[ESTIMATOR_NAMES, FEATURE_SELECTOR_NAMES, SCALER_NAMES, SCORER_NAMES, SEARCHER_NAMES]))

    for estimator, feature_selector, scaler, scorer, searcher in all_pipelines:
        if estimator in IGNORE_ESTIMATOR or\
            feature_selector in IGNORE_FEATURE_SELECTOR or\
            scaler in IGNORE_SCALER or\
            searcher in IGNORE_SEARCHERS or\
            scorer in IGNORE_SCORER:
            continue

        key = '__'.join([scaler, feature_selector, estimator, scorer, searcher])
        print('Generating ' + model_key_to_name(key))

        pipeline = generate_pipeline(scaler, feature_selector, estimator, scorer, searcher)
        models += pipeline[1]
        model = generate_model(estimator, pipeline[0], feature_names, x_train, y_train, scorer)
        results[key] = generalize(model, pipeline[0], x2, y2, labels)
        results[key]['best_params'] = model['best_params']

    print('Total fits generated: %d' % models)
    print_summary(results)
    return results
