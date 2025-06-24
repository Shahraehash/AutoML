"""
Compute precision recall curve and precision score
"""

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import precision_score, recall_score, precision_recall_curve

from .preprocess import preprocess
from .utils import decimate_points

def precision_recall(pipeline, features, model, x_test, y_test, class_index=None):
    """Compute precision recall curve"""

    # Transform values based on the pipeline
    x_test = preprocess(features, pipeline, x_test)

    n_classes = len(np.unique(y_test))

    if hasattr(model, 'decision_function'):
        probabilities = model.decision_function(x_test)
        
        # Binary Classification 
        if n_classes == 2:
            if np.count_nonzero(probabilities):
                if probabilities.max() - probabilities.min() == 0:
                    probabilities = [0] * len(probabilities)
                else:
                    probabilities = (probabilities - probabilities.min()) / \
                        (probabilities.max() - probabilities.min())
            precision, recall, _ = precision_recall_curve(y_test, probabilities)
        
        # Multi-Class Classification - use one-vs-rest approach
        else:
            # If class_index is specified, return One-vs-Rest curve for that class
            if class_index is not None:
                unique_classes = sorted(np.unique(y_test))
                if class_index < len(unique_classes):
                    actual_class_value = unique_classes[class_index]
                    y_binary = (y_test == actual_class_value).astype(int)
                    class_scores = probabilities[:, class_index]
                    precision, recall, _ = precision_recall_curve(y_binary, class_scores)
                else:
                    # Invalid class index, fall back to macro average
                    class_index = None
            
            # If class_index is None or invalid, return macro-averaged curve (original behavior)
            if class_index is None:
                # Compute precision-recall curves for each class and average
                precision_curves = []
                recall_curves = []
                
                for class_idx in range(n_classes):
                    # Create binary labels for current class vs rest
                    y_binary = (y_test == class_idx).astype(int)
                    
                    # Use decision function score for this class
                    class_scores = probabilities[:, class_idx]
                    
                    # Compute precision-recall curve for this class
                    prec_class, rec_class, _ = precision_recall_curve(y_binary, class_scores)
                    precision_curves.append(prec_class)
                    recall_curves.append(rec_class)
                
                # Average the curves (pad to same length first)
                max_len = max(len(curve) for curve in precision_curves)
                precision_padded = []
                recall_padded = []
                
                for i in range(len(precision_curves)):
                    prec_padded = np.pad(precision_curves[i], (0, max_len - len(precision_curves[i])), constant_values=np.nan)
                    rec_padded = np.pad(recall_curves[i], (0, max_len - len(recall_curves[i])), constant_values=np.nan)
                    precision_padded.append(prec_padded)
                    recall_padded.append(rec_padded)
                
                precision = np.nanmean(precision_padded, axis=0)
                recall = np.nanmean(recall_padded, axis=0)
    
    else:

        # Binary classification
        if n_classes == 2:
            probabilities = model.predict_proba(x_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, probabilities)
            
        # Multiclass classification - use one-vs-rest approach
        else:
            probabilities = model.predict_proba(x_test)
            
            # If class_index is specified, return One-vs-Rest curve for that class
            if class_index is not None:
                unique_classes = sorted(np.unique(y_test))
                if class_index < len(unique_classes):
                    actual_class_value = unique_classes[class_index]
                    y_binary = (y_test == actual_class_value).astype(int)
                    class_probs = probabilities[:, class_index]
                    precision, recall, _ = precision_recall_curve(y_binary, class_probs)
                else:
                    # Invalid class index, fall back to macro average
                    class_index = None
            
            # If class_index is None or invalid, return macro-averaged curve (original behavior)
            if class_index is None:
                precision_curves = []
                recall_curves = []
                for class_idx in range(n_classes):
                    # Create binary labels for current class vs rest
                    y_binary = (y_test == class_idx).astype(int)
                    
                    # Use probability for this class
                    class_probs = probabilities[:, class_idx]
                    prec_class, rec_class, _ = precision_recall_curve(y_binary, class_probs)
                    precision_curves.append(prec_class)
                    recall_curves.append(rec_class)
                
                # Average the curves (pad to same length first so we can take the mean of it)
                max_len = max(len(curve) for curve in precision_curves)
                precision_padded = []
                recall_padded = []
                
                for i in range(len(precision_curves)):
                    prec_padded = np.pad(precision_curves[i], (0, max_len - len(precision_curves[i])), constant_values=np.nan)
                    rec_padded = np.pad(recall_curves[i], (0, max_len - len(recall_curves[i])), constant_values=np.nan)
                    precision_padded.append(prec_padded)
                    recall_padded.append(rec_padded)
                
                precision = np.nanmean(precision_padded, axis=0)
                recall = np.nanmean(recall_padded, axis=0)


    recall, precision = decimate_points(
      [round(num, 4) for num in list(recall)],
      [round(num, 4) for num in list(precision)]
    )

    return {
        'precision': list(precision),
        'recall': list(recall)
    }

def additional_precision(payload, label, folder, class_index=None):
    """Return additional precision recall curve"""

    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    x = data[payload['features']].to_numpy()
    y = data[label]

    pipeline = load(folder + '.joblib')

    return precision_recall(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)
