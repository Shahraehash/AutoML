"""
Compute reliability curve and Briar score
"""

import numpy as np
import pandas as pd
from joblib import load

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from .preprocess import preprocess

def _is_binary_classification(y_test, probabilities):
    """
    Determine if this is true binary classification based on both 
    the target values and model output.
    
    True binary classification means the model was naturally trained
    on 2 classes, regardless of class_index parameter.
    """
    n_unique_classes = len(np.unique(y_test))
    n_prob_classes = probabilities.shape[1]
    
    return n_unique_classes == 2 and n_prob_classes == 2


def _compute_binary_reliability(y_test, probabilities):
    """
    Compute reliability metrics for binary classification.
    
    Args:
        y_test: True binary labels
        probabilities: Probability array with shape (n_samples, 2)
    
    Returns:
        tuple: (fop, mpv, brier_score)
    """
    # Use probabilities for positive class (index 1)
    pos_probs = probabilities[:, 1]
    
    # Handle edge case where all probabilities are the same
    if pos_probs.max() - pos_probs.min() == 0:
        pos_probs = np.full_like(pos_probs, 0.5)
    
    fop, mpv = calibration_curve(y_test, pos_probs, n_bins=10, strategy='uniform')
    brier_score = brier_score_loss(y_test, pos_probs)
    
    return fop, mpv, brier_score


def _compute_multiclass_reliability(y_test, probabilities, class_index=None):
    """
    Compute reliability metrics for multi-class classification.
    
    Args:
        y_test: True labels
        probabilities: Probability array with shape (n_samples, n_classes)
        class_index: If specified, compute One-vs-Rest for this class only
    
    Returns:
        tuple: (fop, mpv, brier_score)
    """
    unique_classes = sorted(np.unique(y_test))
    n_classes = len(unique_classes)
    
    # Validate class_index if provided
    if class_index is not None:
        if not isinstance(class_index, int) or class_index < 0 or class_index >= n_classes:
            raise ValueError(f"class_index must be an integer between 0 and {n_classes-1}")
    
    # Case 1: Specific class One-vs-Rest
    if class_index is not None:
        actual_class_value = unique_classes[class_index]
        y_binary = (y_test == actual_class_value).astype(int)
        class_probs = probabilities[:, class_index]
        
        fop, mpv = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
        brier_score = brier_score_loss(y_binary, class_probs)
        
        return fop, mpv, brier_score
    
    # Case 2: Macro-averaged One-vs-Rest across all classes
    fop_list = []
    mpv_list = []
    brier_scores = []
    
    for class_idx in range(n_classes):
        actual_class_value = unique_classes[class_idx]
        y_binary = (y_test == actual_class_value).astype(int)
        class_probs = probabilities[:, class_idx]
        
        # Compute calibration curve for this class
        fop_class, mpv_class = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
        fop_list.append(fop_class)
        mpv_list.append(mpv_class)
        
        # Compute Brier score for this class
        brier_class = brier_score_loss(y_binary, class_probs)
        brier_scores.append(brier_class)
    
    # Handle averaging of arrays with potentially different lengths
    if not fop_list:
        return np.array([]), np.array([]), 0.0
    
    max_len = max(len(arr) for arr in fop_list)
    
    if max_len == 0:
        return np.array([]), np.array([]), np.mean(brier_scores)
    
    # Pad arrays to same length with NaN
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
    brier_score = np.mean(brier_scores)
    
    return fop, mpv, brier_score


def reliability(pipeline, features, model, x_test, y_test, class_index=None):
    """
    Compute reliability curve and Brier score for classification models.
    
    This function handles different classification scenarios:
    - True Binary Classification: Model naturally trained on 2 classes
      (class_index is ignored - uses standard binary evaluation)
    - Multi-class with class_index: One-vs-Rest evaluation for the specified class
    - Multi-class without class_index: Macro-averaged One-vs-Rest across all classes
    
    Args:
        pipeline: Preprocessing pipeline
        features: List of feature names
        model: Trained classification model
        x_test: Test features
        y_test: True test labels
        class_index: Optional. For multi-class models, specifies which class
                    to evaluate using One-vs-Rest approach. Ignored for binary models.
    
    Returns:
        dict: Dictionary containing:
            - 'brier_score': Brier score (lower is better)
            - 'fop': Fraction of positives (observed frequencies)
            - 'mpv': Mean predicted values (predicted probabilities)
    
    Raises:
        ValueError: If class_index is invalid for the given data
    """
    # Validate inputs
    if len(x_test) == 0 or len(y_test) == 0:
        raise ValueError("Test data cannot be empty")
    
    if len(x_test) != len(y_test):
        raise ValueError("x_test and y_test must have the same length")
    
    # Preprocess the test data
    x_test_processed = preprocess(features, pipeline, x_test)
    
    # Extract probabilities from the model (all models have predict_proba)
    probabilities = model.predict_proba(x_test_processed)
    
    # Determine if this is true binary classification (naturally 2 classes)
    is_binary = _is_binary_classification(y_test, probabilities)
    
    if is_binary:
        # True binary classification - ignore class_index parameter
        fop, mpv, brier_score = _compute_binary_reliability(y_test, probabilities)
    else:
        # Multi-class model - respect class_index parameter
        fop, mpv, brier_score = _compute_multiclass_reliability(y_test, probabilities, class_index)
    
    # Return results with consistent formatting
    return {
        'brier_score': round(float(brier_score), 4),
        'fop': [round(float(num), 4) for num in fop],
        'mpv': [round(float(num), 4) for num in mpv]
    }


def additional_reliability(payload, label, folder, class_index=None):
    """
    Compute reliability metrics from a payload and saved model.
    
    Args:
        payload: Dictionary containing data and metadata
        label: Target column name
        folder: Path to saved model (without .joblib extension)
        class_index: Optional class index for multi-class One-vs-Rest
    
    Returns:
        dict: Reliability metrics dictionary
    """
    # Load and prepare data
    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    
    if data.empty:
        raise ValueError("No valid data remaining after preprocessing")
    
    x = data[list(payload['features'])].to_numpy()
    y = data[label].to_numpy()
    
    # Load the trained pipeline
    pipeline = load(folder + '.joblib')
    
    # Extract the model from the pipeline
    model = pipeline.steps[-1][1]
    
    return reliability(pipeline, payload['features'], model, x, y, class_index)
