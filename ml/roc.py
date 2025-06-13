"""
Compute receiver operating characteristic
"""

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

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
        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro') 
        
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        if y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
        

        fpr_per_class = []
        tpr_per_class = []

        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
            fpr_per_class.append(fpr_i)
            tpr_per_class.append(tpr_i)
        
        
        fpr = np.unique(np.concatenate(fpr_per_class))
        tpr = np.zeros_like(fpr)
        
        for i in range(n_classes):
            # Use numpy's interp function instead of scipy
            interp_tpr = np.interp(fpr, fpr_per_class[i], tpr_per_class[i])
            tpr += interp_tpr
        
        tpr /= n_classes
        
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
