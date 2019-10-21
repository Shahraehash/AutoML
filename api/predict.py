"""
Predicts outcome from incoming data against exported model
"""

from joblib import load

def predict(data):
    """Predicts against the provided data"""

    pipeline = load('pipeline.joblib')
    return pipeline.predict(data)
