"""
Compute receiver operating characteristic
"""

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from .preprocess import preprocess
from .utils import decimate_points

def roc(pipeline, features, model, x_test, y_test):
    """Generate the ROC values"""

    # Transform values based on the pipeline
    x_test = preprocess(features, pipeline, x_test)

    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    
    # Check if this is binary or multiclass classification
    n_classes = probabilities.shape[1]
    
    # Binary classification
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
        roc_auc = roc_auc_score(y_test, probabilities[:, 1])

    # Multiclass classification
    else:
        # Perform multi-class as one-vs-rest strategy with macro averaging
        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro') 
        cnf_matrix = confusion_matrix(y_test, predictions)

        fp = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
        fn = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
        tp = (np.diag(cnf_matrix)).astype(float)
        tn = (cnf_matrix.sum() - (fp + fn + tp)).astype(float)       

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        
    fpr, tpr = decimate_points(
      [round(num, 4) for num in list(fpr)],
      [round(num, 4) for num in list(tpr)]
    )

    return {
        'fpr': list(fpr),
        'tpr': list(tpr),
        'roc_auc': roc_auc
    }

def additional_roc(payload, label, folder):
    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    x = data[payload['features']].to_numpy()
    y = data[label]

    pipeline = load(folder + '.joblib')

    return roc(pipeline, payload['features'], pipeline.steps[-1][1], x, y)
