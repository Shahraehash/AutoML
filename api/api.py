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
IGNORE_SCORER = os.getenv('IGNORE_SCORER', '').split(',')

def find_best_model(train_set='sample-data/train.csv',
                    test_set='sample-data/test.csv', labels=None, label_column='AKI'):
    """Generates all possible models and outputs the generalization results"""

    # Define the labels for our classes
    # This is used for the classification reproting
    if labels is None:
        labels = ['No AKI', 'AKI']

    # Import data
    (x_train, y_train, x2, y2, feature_names) = import_data(train_set, test_set, label_column)

    # Generate all models
    results = {}

    all_pipelines = list(itertools.product(
        *[ESTIMATOR_NAMES, FEATURE_SELECTOR_NAMES, SCALER_NAMES, SCORER_NAMES]))

    for estimator, feature_selector, scaler, scorer in all_pipelines:
        if estimator in IGNORE_ESTIMATOR or\
            feature_selector in IGNORE_FEATURE_SELECTOR or\
            scaler in IGNORE_SCALER or\
            scorer in IGNORE_SCORER:
            continue

        key = '__'.join([scaler, feature_selector, estimator, scorer])
        print('Generating ' + model_key_to_name(key))

        pipeline = generate_pipeline(scaler, feature_selector, estimator, scorer)

        results[key] = generalize(
            generate_model(
                estimator, pipeline, feature_names, x_train, y_train, scorer
            ),
            pipeline, x2, y2, labels)

    print_summary(results)
    return results