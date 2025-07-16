"""
Creates the model based on the pipeline components (provided by
the key, hyper parameters and features used).
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import dump, load
from nyoka import skl_to_pmml, xgboost_to_pmml
from sklearn.pipeline import Pipeline

from .processors.estimators import ESTIMATORS, get_xgb_classifier
from .processors.feature_selection import FEATURE_SELECTORS
from .processors.scalers import SCALERS
from .import_data import import_data
from .generalization import generalize
from .model import generate_model
from .utils import explode_key

def create_model(key, hyper_parameters, selected_features, dataset_path=None, label_column=None, output_path='.', threshold=.5):
    """Refits the requested model and pickles it for export"""

    if dataset_path is None:
        print('Missing dataset path')
        return {}

    if label_column is None:
        print('Missing column name for classifier target')
        return {}

    # Import data
    (x_train, _, y_train, _, x2, y2, features, metadata) = \
        import_data(dataset_path + '/train.csv', dataset_path + '/test.csv', label_column)
    
    # Extract label mapping information
    label_mapping_info = metadata.get('label_mapping')

    # Get pipeline details from the key
    scaler, feature_selector, estimator, _, _ = explode_key(key)
    steps = []

    # Drop the unused features
    if 'pca-' not in feature_selector:
        for index, feature in reversed(list(enumerate(features))):
            if feature not in selected_features:
                x_train = np.delete(x_train, index, axis=1)
                x2 = np.delete(x2, index, axis=1)

    # Add the scaler, if used
    if scaler and SCALERS[scaler]:
        steps.append(('scaler', SCALERS[scaler]))

    # Add the feature transformer
    if 'pca-' in feature_selector:
        steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

    # Add the estimator with proper XGBoost configuration
    if estimator == 'gb':
        n_classes = len(pd.Series(y_train).unique())
        base_estimator = get_xgb_classifier(n_classes)
    else:
        base_estimator = ESTIMATORS[estimator]
    
    steps.append(('estimator', base_estimator.set_params(**hyper_parameters)))

    # Fit the pipeline using the same training data
    pipeline = Pipeline(steps)
    model = generate_model(pipeline, selected_features, x_train, y_train)

    # If the model is DNN or RF, attempt to swap the estimator for a pickled one
    if os.path.exists(output_path + '/models/' + key + '.joblib'):
      pickled_estimator = load(output_path + '/models/' + key + '.joblib')
      pipeline = Pipeline(pipeline.steps[:-1] + [('estimator', pickled_estimator)])

    unique_labels = sorted(y2.unique())
    
    # Generate labels based on original classes if mapping exists
    if label_mapping_info and 'original_labels' in label_mapping_info:
        original_labels = label_mapping_info['original_labels']
        if len(original_labels) == 2:
            labels = ['No ' + label_column, label_column]
        else:
            labels = [f'Class {int(cls)}' for cls in original_labels]
    else:
        # Fallback to current logic for backward compatibility
        if len(unique_labels) == 2:
            labels = ['No ' + label_column, label_column]
        else:
            labels = [f'Class {int(cls)}' for cls in unique_labels]

    # Assess the model performance and store the results
    generalization_result = generalize(pipeline, model['features'], pipeline['estimator'], x2, y2, labels, threshold)
    
    # Add label mapping information to the results
    if label_mapping_info:
        generalization_result['label_mapping'] = label_mapping_info
    
    with open(output_path + '/pipeline.json', 'w') as statsfile:
        json.dump(generalization_result, statsfile)

    # Dump the pipeline to a file
    dump(pipeline, output_path + '/pipeline.joblib')
    pd.DataFrame([selected_features]).to_csv(output_path + '/input.csv', index=False, header=False)

    # Export the model as a PMML
    try:
        if estimator == 'gb':
            xgboost_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
        else:
            skl_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
    except Exception:
        try:
            os.remove(output_path + '/pipeline.pmml')
        except OSError:
            pass

    return generalization_result
