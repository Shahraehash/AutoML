"""
Predicts outcome from incoming data against exported model
"""

from joblib import load
import pandas as pd
import numpy as np
import json

def predict(data, path='.', threshold=.5):
    """Predicts against the provided data"""

    # Load the pipeline
    pipeline = load(path + '.joblib')
    
    # Load label mapping if it exists
    label_mapping_info = None
    try:
        with open(path + '.json', 'r') as f:
            pipeline_info = json.load(f)
            label_mapping_info = pipeline_info.get('label_mapping')
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    data = pd.DataFrame(data).dropna().values

    probability = pipeline.predict_proba(data)
    if threshold == .5:
      predicted = pipeline.predict(data)
    else:
      predicted = (probability[:, 1] >= threshold).astype(int)

    # Map predictions back to original labels if mapping exists
    if label_mapping_info and 'inverse_mapping' in label_mapping_info:
        inverse_mapping = label_mapping_info['inverse_mapping']
        # Convert keys to integers for mapping
        inverse_mapping = {int(k): v for k, v in inverse_mapping.items()}
        predicted_original = [inverse_mapping.get(int(pred), pred) for pred in predicted]
    else:
        predicted_original = predicted.tolist()

    return {
        'predicted': predicted_original,
        'probability': [sublist[predicted[index]] for index, sublist in enumerate(probability.tolist())]
    }

def predict_ensemble(total_models, data, path='.', vote_type='soft'):
    """Predicts against the provided data by creating an ensemble of the selected models"""

    probabilities = []
    predictions = []
    label_mapping_info = None

    for x in range(total_models):
        pipeline = load(path + '/ensemble' + str(x) + '.joblib')

        with open(path + '/ensemble' + str(x) + '_features.json') as feature_file:
            features = json.load(feature_file)

        # Load label mapping from the first model (should be consistent across ensemble)
        if x == 0 and label_mapping_info is None:
            try:
                with open(path + '/ensemble' + str(x) + '.json', 'r') as f:
                    pipeline_info = json.load(f)
                    label_mapping_info = pipeline_info.get('label_mapping')
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        selected_data = data[features].dropna().to_numpy()
        probabilities.append(pipeline.predict_proba(selected_data))
        predictions.append(pipeline.predict(selected_data))

    predictions = np.asarray(predictions).T
    probabilities = np.average(np.asarray(probabilities), axis=0)

    if vote_type == 'soft':
        predicted = np.argmax(probabilities, axis=1)
    else:
        predicted = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions.astype('int')
        )

    # Map predictions back to original labels if mapping exists
    if label_mapping_info and 'inverse_mapping' in label_mapping_info:
        inverse_mapping = label_mapping_info['inverse_mapping']
        # Convert keys to integers for mapping
        inverse_mapping = {int(k): v for k, v in inverse_mapping.items()}
        predicted_original = [inverse_mapping.get(int(pred), pred) for pred in predicted]
    else:
        predicted_original = predicted.tolist()

    return {
        'predicted': predicted_original,
        'probability': [sublist[predicted[index]] for index, sublist in enumerate(probabilities.tolist())]
    }
