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
    
    # Get unique classes and create consistent mapping
    unique_classes = sorted(np.unique(y_test))
    n_classes = len(unique_classes)
    
    # Create a mapping from class values to indices
    class_to_idx = {class_val: idx for idx, class_val in enumerate(unique_classes)}
    
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(x_test)
        
        # Binary Classification 
        if n_classes == 2:
            # For binary classification, decision_function returns 1D array
            if scores.ndim == 1:
                # Use scores directly - they're already decision function values
                precision, recall, _ = precision_recall_curve(y_test, scores)
            else:
                # Some classifiers might return 2D even for binary
                precision, recall, _ = precision_recall_curve(y_test, scores[:, 1])
        
        # Multi-Class Classification
        else:
            if class_index is not None and 0 <= class_index < n_classes:
                # One-vs-Rest for specific class
                actual_class_value = unique_classes[class_index]
                y_binary = (y_test == actual_class_value).astype(int)
                class_scores = scores[:, class_index]
                precision, recall, _ = precision_recall_curve(y_binary, class_scores)
            else:
                # Macro-averaged curve
                precision, recall = compute_macro_averaged_curve(
                    y_test, scores, unique_classes, use_proba=False
                )
    
    else:
        # Use predict_proba (Random Forest, etc.)
        probabilities = model.predict_proba(x_test)
        
        # Binary classification
        if n_classes == 2:
            # Use probability of positive class
            precision, recall, _ = precision_recall_curve(y_test, probabilities[:, 1])
            
        # Multiclass classification
        else:
            if class_index is not None and 0 <= class_index < n_classes:
                # One-vs-Rest for specific class
                actual_class_value = unique_classes[class_index]
                y_binary = (y_test == actual_class_value).astype(int)
                class_probs = probabilities[:, class_index]
                precision, recall, _ = precision_recall_curve(y_binary, class_probs)
            else:
                # Macro-averaged curve
                precision, recall = compute_macro_averaged_curve(
                    y_test, probabilities, unique_classes, use_proba=True
                )

    # Apply decimation
    recall, precision = decimate_points(
        [round(num, 4) for num in list(recall)],
        [round(num, 4) for num in list(precision)]
    )

    return {
        'precision': list(precision),
        'recall': list(recall)
    }

def compute_macro_averaged_curve(y_test, scores_or_probs, unique_classes, use_proba=True):
    """Helper function to compute macro-averaged precision-recall curve"""
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    from scipy import interpolate
    
    # Use common recall points for interpolation
    common_recall = np.linspace(0, 1, 101)  # 101 points from 0 to 1
    precision_interp_curves = []
    
    for class_idx, class_val in enumerate(unique_classes):
        # Create binary labels for current class vs rest
        y_binary = (y_test == class_val).astype(int)
        
        # Skip if no positive samples for this class
        if y_binary.sum() == 0:
            continue
            
        # Get scores for this class
        class_scores = scores_or_probs[:, class_idx]
        
        # Compute precision-recall curve
        prec_class, rec_class, _ = precision_recall_curve(y_binary, class_scores)
        
        # Interpolate to common recall points
        # Note: precision_recall_curve returns decreasing recall, so we reverse
        prec_class = prec_class[::-1]
        rec_class = rec_class[::-1]
        
        # Interpolate precision at common recall points
        prec_interp = np.interp(common_recall, rec_class, prec_class)
        precision_interp_curves.append(prec_interp)
    
    if not precision_interp_curves:
        # Fallback if no valid curves
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    
    # Average the interpolated curves
    precision_avg = np.mean(precision_interp_curves, axis=0)
    
    return precision_avg, common_recall

def additional_precision(payload, label, folder, class_index=None):
    """Return additional precision recall curve"""

    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    x = data[payload['features']].to_numpy()
    y = data[label]

    pipeline = load(folder + '.joblib')

    return precision_recall(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)
