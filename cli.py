"""
AutoML

Interact with the trainer using the CLI
"""
import os
import sys
import pandas as pd

from ml import search

# Parse input or load sample data
if len(sys.argv) < 4:
    TRAIN_SET = 'sample-data/train.csv'
    TEST_SET = 'sample-data/test.csv'
    LABEL_COLUMN = 'Cancer'
else:
    TRAIN_SET = sys.argv[1]
    TEST_SET = sys.argv[2]
    LABEL_COLUMN = sys.argv[3]

data = pd.read_csv(TRAIN_SET)
unique_labels = sorted(data[LABEL_COLUMN].dropna().unique())

# Binary classification
if len(unique_labels) == 2: 
    LABELS = ['No ' + LABEL_COLUMN, LABEL_COLUMN]
# Multiclass classification
else:
    LABELS = [f'{label_column}_class_{int(label)}' for label in unique_labels]

PARAMETERS = dict(
    ignore_estimator=os.getenv('IGNORE_ESTIMATOR', ''),
    ignore_feature_selector=os.getenv('IGNORE_FEATURE_SELECTOR', ''),
    ignore_scaler=os.getenv('IGNORE_SCALER', ''),
    ignore_searcher=os.getenv('IGNORE_SEARCHER', ''),
    ignore_shuffle=os.getenv('IGNORE_SHUFFLE', ''),
    ignore_scorer=os.getenv('IGNORE_SCORER', '')
)

hyper_parameters = os.getenv('CUSTOM_HYPER_PARAMETERS', None)

if hyper_parameters is not None:
    PARAMETERS['hyper_parameters'] = hyper_parameters

search.find_best_model(TRAIN_SET, TEST_SET, LABELS, LABEL_COLUMN, PARAMETERS)
