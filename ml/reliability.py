"""
Compute reliability curve and Briar score
"""

import numpy as np
import pandas as pd
from joblib import load

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from .preprocess import preprocess

def reliability(pipeline, features, model, x_test, y_test, class_index=None):
    """Compute reliability curve and Briar score"""

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
            fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')
            brier_score = brier_score_loss(y_test, probabilities)
        
        # Multi-Class Classification
        else:
            # Use softmax to convert decision function scores to probabilities
            exp_scores = np.exp(probabilities - np.max(probabilities, axis=1, keepdims=True))
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # If class_index is specified, return One-vs-Rest curve for that class
            if class_index is not None:
                unique_classes = sorted(np.unique(y_test))
                if class_index < len(unique_classes):
                    actual_class_value = unique_classes[class_index]
                    y_binary = (y_test == actual_class_value).astype(int)
                    class_probs = probabilities[:, class_index]
                    fop, mpv = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
                    brier_score = brier_score_loss(y_binary, class_probs)
                else:
                    # Invalid class index, fall back to macro average
                    class_index = None
            
            # If class_index is None or invalid, return macro-averaged curve (original behavior)
            if class_index is None:
                # Use one-vs-rest approach for calibration curves
                fop_list = []
                mpv_list = []
                brier_scores = []
                for class_idx in range(n_classes):
                    # Create binary labels for current class vs rest
                    y_binary = (y_test == class_idx).astype(int)
                    class_probs = probabilities[:, class_idx]
                    
                    # Compute calibration curve for this class
                    fop_class, mpv_class = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
                    fop_list.append(fop_class)
                    mpv_list.append(mpv_class)
                    
                    # Compute Brier score for this class
                    brier_class = brier_score_loss(y_binary, class_probs)
                    brier_scores.append(brier_class)
                
                # Ensure all arrays have the same length by padding with NaN and then averaging
                max_len = max(len(arr) for arr in fop_list) if fop_list else 0
                if max_len > 0:
                    # Pad arrays to same length
                    fop_padded = []
                    mpv_padded = []
                    for i in range(len(fop_list)):
                        fop_arr = np.pad(fop_list[i], (0, max_len - len(fop_list[i])), constant_values=np.nan)
                        mpv_arr = np.pad(mpv_list[i], (0, max_len - len(mpv_list[i])), constant_values=np.nan)
                        fop_padded.append(fop_arr)
                        mpv_padded.append(mpv_arr)
                    
                    # Average ignoring NaN values
                    fop = np.nanmean(fop_padded, axis=0)
                    mpv = np.nanmean(mpv_padded, axis=0)
                else:
                    fop = np.array([])
                    mpv = np.array([])
                brier_score = np.mean(brier_scores)
    
    else:

        # Binary classification
        if n_classes == 2:
            probabilities = model.predict_proba(x_test)[:, 1]
            fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')
            brier_score = brier_score_loss(y_test, probabilities)
            
        # Multi-class Classification
        else:
            probabilities = model.predict_proba(x_test)
        
            # If class_index is specified, return One-vs-Rest curve for that class
            if class_index is not None:
                unique_classes = sorted(np.unique(y_test))
                if class_index < len(unique_classes):
                    actual_class_value = unique_classes[class_index]
                    y_binary = (y_test == actual_class_value).astype(int)
                    class_probs = probabilities[:, class_index]
                    fop, mpv = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
                    brier_score = brier_score_loss(y_binary, class_probs)
                else:
                    # Invalid class index, fall back to macro average
                    class_index = None
            
            # If class_index is None or invalid, return macro-averaged curve (original behavior)
            if class_index is None:
                # Use one-vs-rest approach for calibration curves
                fop_list = []
                mpv_list = []
                brier_scores = []
                
                for class_idx in range(n_classes):
                    # Create binary labels for current class vs rest
                    y_binary = (y_test == class_idx).astype(int)
                    class_probs = probabilities[:, class_idx]
                    
                    # Compute calibration curve for this class
                    fop_class, mpv_class = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
                    fop_list.append(fop_class)
                    mpv_list.append(mpv_class)
                    
                    # Compute Brier score for this class
                    brier_class = brier_score_loss(y_binary, class_probs)
                    brier_scores.append(brier_class)
                
                # Ensure all arrays have the same length by padding with NaN and then averaging
                max_len = max(len(arr) for arr in fop_list) if fop_list else 0
               
                fop_padded = []
                mpv_padded = []
                for i in range(len(fop_list)):
                    fop_arr = np.pad(fop_list[i], (0, max_len - len(fop_list[i])), constant_values=np.nan)
                    mpv_arr = np.pad(mpv_list[i], (0, max_len - len(mpv_list[i])), constant_values=np.nan)
                    fop_padded.append(fop_arr)
                    mpv_padded.append(mpv_arr)
                
                fop = np.nanmean(fop_padded, axis=0)
                mpv = np.nanmean(mpv_padded, axis=0)
                brier_score = np.mean(brier_scores)

    return {
        'brier_score': round(brier_score, 4),
        'fop': [round(num, 4) for num in list(fop)],
        'mpv': [round(num, 4) for num in list(mpv)]
    }

def additional_reliability(payload, label, folder, class_index=None):
    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    x = data[payload['features']].to_numpy()
    y = data[label]

    pipeline = load(folder + '.joblib')

    return reliability(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)
