"""
Auto ML

Supervised learning using an exhaustive search of ideal pre-processing (if any), algorithms,
and hyper-parameters with feature engineering.
"""

# Dependencies
import os
import itertools
import sys

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

# Parse input or load sample data
if len(sys.argv) < 3:
    TRAIN_SET = 'sample-data/train.csv'
    TEST_SET = 'sample-data/test.csv'
else:
    TRAIN_SET = sys.argv[1]
    TEST_SET = sys.argv[2]

# Define the labels for our classes
# This is used for the classification reproting (more readable then 0/1)
LABELS = ['No AKI', 'AKI']
LABEL_COLUMN = 'AKI'

# Import data
(X_TRAIN, Y_TRAIN, X2, Y2, FEATURE_NAMES) = import_data(TRAIN_SET, TEST_SET, LABEL_COLUMN)

# Generate all models
RESULTS = {}

ALL_PIPELINES = list(itertools.product(
    *[ESTIMATOR_NAMES, FEATURE_SELECTOR_NAMES, SCALER_NAMES, SCORER_NAMES]))

for estimator, feature_selector, scaler, scorer in ALL_PIPELINES:
    if estimator in IGNORE_ESTIMATOR or\
        feature_selector in IGNORE_FEATURE_SELECTOR or\
        scaler in IGNORE_SCALER or\
        scorer in IGNORE_SCORER:
        continue

    key = '__'.join([scaler, feature_selector, estimator, scorer])
    print('Generating ' + model_key_to_name(key))

    pipeline = generate_pipeline(scaler, feature_selector, estimator, scorer)

    RESULTS[key] = generalize(
        generate_model(
            estimator, pipeline, FEATURE_NAMES, X_TRAIN, Y_TRAIN, scorer
        ),
        pipeline, X2, Y2, LABELS)

print_summary(RESULTS)
