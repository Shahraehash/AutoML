"""
Predicts outcome from incoming data against exported model
"""

from joblib import load

def predict(data, train_set=None, label_column=None):
    """Predicts against the provided data"""

    if train_set is None:
        print('Missing training data')
        return {}

    if label_column is None:
        print('Missing column name for classifier target')
        return {}

    # Load the pipeline
    pipeline = load('pipeline.joblib')

    predicted = pipeline.predict([data])
    probability = pipeline.predict_proba([data])[:, predicted]
    print(probability)

    return {
        'predicted': int(predicted[0]),
        'probability': probability[0][0]
    }
