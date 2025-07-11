"""
Refit a model based on the provided parameters
and score it against the test set for the provided
scoring method.
"""

import json
import pandas as pd
from sklearn.base import clone

from .processors.estimators import ESTIMATORS, get_xgb_classifier
from .processors.scorers import SCORER_NAMES
from .preprocess import preprocess

MODELS_TO_EVALUATE = 2

def filter_params_for_binary_classification(params, estimator, n_classes):
    """
    Filter hyperparameters to ensure compatibility with binary classification.
    
    Args:
        params: Dictionary of hyperparameters from search results
        estimator: Estimator type ('gb', 'lr', etc.)
        n_classes: Number of unique classes in current y_train
    
    Returns:
        Filtered parameter dictionary safe for binary classification
    """
    filtered_params = params.copy()
    
    # Handle XGBoost binary classification
    if estimator == 'gb' and n_classes == 2:
        # Remove or fix multiclass-specific parameters
        if 'objective' in filtered_params:
            if filtered_params['objective'] in ['multi:softprob', 'multi:softmax']:
                # Replace with binary objective
                filtered_params['objective'] = 'binary:logistic'
        
        # Remove num_class if present (not needed for binary)
        if 'num_class' in filtered_params:
            del filtered_params['num_class']
            
        # Ensure eval_metric is appropriate for binary
        if 'eval_metric' in filtered_params:
            if filtered_params['eval_metric'] == 'mlogloss':
                filtered_params['eval_metric'] = 'logloss'
    
    return filtered_params

def refit_model(pipeline, features, estimator, scoring, x_train, y_train):
    """
    Determine the best model based on the provided scoring method
    and the fitting pipeline results.
    """

    # Transform values based on the pipeline
    x_train = preprocess(features, pipeline, x_train)
    
    # Validation: Ensure we have valid class count
    n_classes = len(pd.Series(y_train).unique())
    if n_classes < 2:
        raise ValueError(f"Invalid number of classes: {n_classes}. Need at least 2 classes for classification.")

    results = pipeline.named_steps['estimator'].cv_results_

    # Select the top search results
    sorted_results = sorted(
        range(len(results['rank_test_%s' % scoring])),
        key=lambda i: results['rank_test_%s' % scoring][i]
    )[:MODELS_TO_EVALUATE]

    models = []

    for position, index in enumerate(sorted_results):
        best_params_ = results['params'][index]

        print('\t#%d %s: %.7g (sd=%.7g)'
              % (position+1, SCORER_NAMES[scoring], results['mean_test_%s' % scoring][index],
                 results['std_test_%s' % scoring][index]))
        print('\t#%d %s parameters:' % (position+1, SCORER_NAMES[scoring]),
              json.dumps(best_params_, indent=4, sort_keys=True).replace('\n', '\n\t'))

        # Handle XGBoost configuration properly
        if estimator == 'gb':
            n_classes = len(pd.Series(y_train).unique())
            base_estimator = get_xgb_classifier(n_classes)
        else:
            base_estimator = ESTIMATORS[estimator]
            n_classes = len(pd.Series(y_train).unique())

        # Filter parameters for compatibility with current classification type
        filtered_params = filter_params_for_binary_classification(
            best_params_, estimator, n_classes
        )
        
        # Debug logging for troubleshooting XGBoost binary classification
        if estimator == 'gb' and n_classes == 2:
            print(f"\tBinary classification detected. Original params: {best_params_}")
            print(f"\tFiltered params: {filtered_params}")

        model = clone(base_estimator).set_params(
            **filtered_params).fit(x_train, y_train)

        models.append({
            'best_estimator': model,
            'best_params': filtered_params
        })

    return models
