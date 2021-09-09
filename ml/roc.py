"""
Compute receiver operating characteristic
"""

from sklearn.metrics import roc_curve, roc_auc_score

from .preprocess import preprocess

def roc(pipeline, features, model, x_test, y_test):
    """Generate the ROC values"""

    # Transform values based on the pipeline
    x_test = preprocess(features, pipeline, x_test)

    probabilities = model.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])

    return {
        'fpr': list(fpr),
        'tpr': list(tpr),
        'roc_auc': roc_auc_score(y_test, probabilities[:, 1])
    }
