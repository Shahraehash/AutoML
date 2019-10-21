"""
Predicts outcome from incoming data against exported model
"""

from joblib import load
import numpy as np

from .import_data import import_train

def predict(data, selected_features, train_set=None, label_column=None):
    """Predicts against the provided data"""

    if train_set is None:
        print('Missing training data')
        return {}

    if label_column is None:
        print('Missing column name for classifier target')
        return {}

    # Load the pipeline
    pipeline = load('pipeline.joblib')

    # Reconstruct the original array as required for
    # the scaler
    _, _, _, _, features = import_train(train_set, label_column)
    expanded_data = [0] * len(features)

    for index, feature in list(enumerate(features)):
        if feature in selected_features:
            expanded_data[index] = data[selected_features.index(feature)]
        else:
            expanded_data[index] = 0

    if 'scaler' in pipeline.named_steps:
        expanded_data = pipeline.named_steps['scaler'].transform([expanded_data])

    if 'feature_selector' in pipeline.named_steps:
        expanded_data = pipeline.named_steps['feature_selector'].transform(expanded_data)

    # Or, remove the unused features from the scaled data
    else:
        for index, feature in reversed(list(enumerate(features))):
            if feature not in selected_features:
                expanded_data = np.delete(expanded_data, index, axis=1)

    predicted = pipeline.named_steps['estimator'].predict(expanded_data)
    probability = pipeline.named_steps['estimator'].predict_proba(expanded_data)[:, 1]

    return {
        'predicted': int(predicted[0]),
        'probability': probability[0]
    }
