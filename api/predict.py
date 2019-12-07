"""
Predicts outcome from incoming data against exported model
"""

from joblib import load

def predict(data, path='.'):
    """Predicts against the provided data"""

    # Load the pipeline
    pipeline = load(path + '/pipeline.joblib')

    predicted = pipeline.predict([data])
    probability = pipeline.predict_proba([data])[:, predicted]

    return {
        'predicted': int(predicted[0]),
        'probability': str(probability[0][0])
    }
