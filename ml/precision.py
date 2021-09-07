"""
Compute precision recall curve and precision score
"""

import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_curve

from .preprocess import preprocess

def precision_recall(pipeline, features, model, x_test, y_test):
    """Compute reliability curve and Briar score"""

    # Transform values based on the pipeline
    x_test = preprocess(features, pipeline, x_test)

    if hasattr(model, 'decision_function'):
        probabilities = model.decision_function(x_test)

        if np.count_nonzero(probabilities):
            if probabilities.max() - probabilities.min() == 0:
                probabilities = [0] * len(probabilities)
            else:
                probabilities = (probabilities - probabilities.min()) / \
                    (probabilities.max() - probabilities.min())
    else:
        probabilities = model.predict_proba(x_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    precision_score = average_precision_score(y_test, probabilities)

    return {
        'precision_score': round(precision_score, 4),
        'precision': list(precision),
        'recall': list(recall)
    }
