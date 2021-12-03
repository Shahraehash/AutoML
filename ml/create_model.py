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

from .processors.estimators import ESTIMATORS
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
    (x_train, _, y_train, _, x2, y2, features, _) = \
        import_data(dataset_path + '/train.csv', dataset_path + '/test.csv', label_column)

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

    # Add the estimator
    steps.append(('estimator', ESTIMATORS[estimator].set_params(**hyper_parameters)))

    # Fit the pipeline using the same training data
    pipeline = Pipeline(steps)
    model = generate_model(pipeline, selected_features, x_train, y_train)

    # If the model is DNN or RF, attempt to swap the estimator for a pickled one
    if os.path.exists(output_path + '/models/' + key + '.joblib'):
      pickled_estimator = load(output_path + '/models/' + key + '.joblib')
      pipeline = Pipeline(pipeline.steps[:-1] + [('estimator', pickled_estimator)])

    # Assess the model performance and store the results
    generalization_result = generalize(model['features'], pipeline['estimator'], pipeline, x2, y2, ['No ' + label_column, label_column], threshold)
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
