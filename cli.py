"""
AutoML

Interact with the trainer using the CLI
"""
import os
import sys

from ml import search

# Parse input or load sample data
if len(sys.argv) < 4:
    TRAIN_SET = 'sample-data/train.csv'
    TEST_SET = 'sample-data/test.csv'
    LABEL_COLUMN = 'AKI'
else:
    TRAIN_SET = sys.argv[1]
    TEST_SET = sys.argv[2]
    LABEL_COLUMN = sys.argv[3]

LABELS = ['No ' + LABEL_COLUMN, LABEL_COLUMN]

PARAMETERS = dict(
    ignore_estimator=os.getenv('IGNORE_ESTIMATOR', ''),
    ignore_feature_selector=os.getenv('IGNORE_FEATURE_SELECTOR', ''),
    ignore_scaler=os.getenv('IGNORE_SCALER', ''),
    ignore_searcher=os.getenv('IGNORE_SEARCHER', ''),
    ignore_shuffle=os.getenv('IGNORE_SHUFFLE', ''),
    ignore_scorer=os.getenv('IGNORE_SCORER', ''),
    hyper_parameters=os.getenv('CUSTOM_HYPER_PARAMETERS', '')
)

search.find_best_model(TRAIN_SET, TEST_SET, LABELS, LABEL_COLUMN, PARAMETERS)
