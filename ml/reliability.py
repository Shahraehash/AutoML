"""
Compute reliability curve and Briar score
"""

import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from .preprocess import preprocess
from .utils import decimate_points

def reliability(pipeline, features, model, x_test, y_test):
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

    fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')

    mpv, fop = decimate_points(
      [round(num, 4) for num in list(mpv)],
      [round(num, 4) for num in list(fop)]
    )

    brier_score = brier_score_loss(y_test, probabilities)

    return {
        'brier_score': round(brier_score, 4),
        'fop': list(fop),
        'mpv': list(mpv)
    }
