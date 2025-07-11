"""
Auto ML

Supervised learning using an exhaustive search of ideal pre-processing (if any), algorithms,
and hyper-parameters with feature engineering.
"""

import csv
import json
import time
import itertools
import numpy as np
import tarfile
import tempfile
import os

from dotenv import load_dotenv
from joblib import dump
from timeit import default_timer as timer
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ml.processors.estimators import ESTIMATOR_NAMES
from ml.processors.feature_selection import FEATURE_SELECTOR_NAMES
from ml.processors.scalers import SCALER_NAMES
from ml.processors.searchers import SEARCHER_NAMES
from ml.processors.scorers import SCORER_NAMES
from ml.utils.import_data import import_data
from ml.utils.summary import print_summary

# Import new class-based classifiers
from ml.binary_classifier import BinaryClassifier
from ml.multiclass_macro_classifier import MulticlassMacroClassifier
from ml.multiclass_ovr_classifier import OvRClassifier

# Load environment variables
load_dotenv()

def find_best_model(
        train_set=None,
        test_set=None,
        labels=None,
        label_column=None,
        parameters=None,
        output_path='.',
        update_function=lambda x, y: None
    ):
    """Generates all possible models and outputs the generalization results using new class-based classifiers"""

    # Basic validation
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
    (x_train, x_val, y_train, y_val, x_test, y_test, feature_names, metadata) = \
        import_data(train_set, test_set, label_column)
    
    # Determine number of classes to choose appropriate classifier
    n_classes = len(np.unique(y_train))
    print(f'Detected {n_classes} classes in target column')
    
    # Choose classifier based on number of classes and parameters
    reoptimize_ovr = parameters.get('reoptimize_ovr', 'false').lower() == 'true'
    
    if n_classes == 2:
        print('Using BinaryClassifier for 2-class problem')
        classifier = BinaryClassifier(parameters, output_path, update_function)
    elif n_classes > 2 and reoptimize_ovr:
        print('Using OvRClassifier for multiclass problem with OvR re-optimization')
        classifier = OvRClassifier(parameters, output_path, update_function)
    elif n_classes > 2:
        print('Using MulticlassMacroClassifier for multiclass problem')
        classifier = MulticlassMacroClassifier(parameters, output_path, update_function)
    else:
        print(f'Invalid number of classes: {n_classes}')
        return False
    
    # Train the classifier with correct parameters
    try:
        success = classifier.find_best_model(
            train_set=train_set,
            test_set=test_set,
            labels=labels,
            label_column=label_column,
            parameters=parameters,
            output_path=output_path,
            update_function=update_function
        )
        print(f"Finished the fit {success}")
        if success:
            print_summary(output_path + '/report.csv')
            return True
        else:
            print('Training failed')
            return False
            
    except Exception as e:
        print(f'Error during training: {str(e)}')
        return False
