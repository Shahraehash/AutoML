"""
Compute precision recall curve and precision score
"""

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import precision_recall_curve

from .preprocess import preprocess
from .utils import decimate_points

def precision_recall(pipeline, features, model, x_test, y_test):
    """Compute precision recall curve"""

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

    recall, precision = decimate_points(
      [round(num, 4) for num in list(recall)],
      [round(num, 4) for num in list(precision)]
    )

    return {
        'precision': list(precision),
        'recall': list(recall)
    }

def additional_precision(payload, label, folder):
    """Return additional precision recall curve"""

    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    x = data[payload['features']].to_numpy()
    y = data[label]

    pipeline = load(folder + '.joblib')

    return precision_recall(pipeline, payload['features'], pipeline.steps[-1][1], x, y)
