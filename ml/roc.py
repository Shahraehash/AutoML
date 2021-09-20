"""
Compute receiver operating characteristic
"""

from sklearn.metrics import roc_curve, roc_auc_score

from .preprocess import preprocess
from .utils import decimate_points

def roc(pipeline, features, model, x_test, y_test):
    """Generate the ROC values"""

    # Transform values based on the pipeline
    x_test = preprocess(features, pipeline, x_test)

    probabilities = model.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])

    fpr, tpr = decimate_points(
      [round(num, 4) for num in list(fpr)],
      [round(num, 4) for num in list(tpr)]
    )

    return {
        'fpr': list(fpr),
        'tpr': list(tpr),
        'roc_auc': roc_auc_score(y_test, probabilities[:, 1])
    }
