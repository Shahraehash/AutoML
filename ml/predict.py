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

    data = pd.DataFrame(data).dropna().values

    probability = pipeline.predict_proba(data)
    predicted = (probability[:,1] >= threshold).astype(int)

    return {
        'predicted': predicted.tolist(),
        'probability': [sublist[predicted[index]] for index, sublist in enumerate(probability.tolist())]
    }

def predict_ensemble(total_models, data, path='.', vote_type='soft'):
    """Predicts against the provided data by creating an ensemble of the selected models"""

    probabilities = []
    predictions = []

    for x in range(total_models):
        pipeline = load(path + '/ensemble' + str(x) + '.joblib')

        with open(path + '/ensemble' + str(x) + '_features.json') as feature_file:
            features = json.load(feature_file)

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

    return {
        'predicted': predicted.tolist(),
        'probability': [sublist[predicted[index]] for index, sublist in enumerate(probabilities.tolist())]
    }
