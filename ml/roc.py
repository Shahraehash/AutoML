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


def _compute_binary_roc(y_test, probabilities):
    """
    Compute ROC curve for binary classification.
    
    Args:
        y_test: True labels (may be multi-class for OvR scenarios)
        probabilities: Probability array with shape (n_samples, 2)
    
    Returns:
        tuple: (fpr, tpr, roc_auc)
    """
    # Use probabilities for positive class (index 1)
    pos_probs = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
    
    # Handle OvR scenario: if y_test has more than 2 classes but we have binary probabilities,
    # we need to convert y_test to binary format for proper evaluation
    unique_classes = sorted(np.unique(y_test))
    if len(unique_classes) > 2:
        # This is an OvR scenario - convert to binary based on the most frequent class
        most_frequent_class = max(set(y_test), key=list(y_test).count)
        y_test_binary = (y_test == most_frequent_class).astype(int)
    else:
        # True binary classification
        y_test_binary = y_test
    
    # Compute ROC curve using binary labels
    fpr, tpr, _ = roc_curve(y_test_binary, pos_probs)
    roc_auc = roc_auc_score(y_test_binary, pos_probs)
    
    return fpr, tpr, roc_auc


def _compute_multiclass_roc(y_test, probabilities, class_index=None):
    """
    Compute ROC curve for multi-class classification.
    
    Args:
        y_test: True labels
        probabilities: Probability array with shape (n_samples, n_classes)
        class_index: If specified, compute One-vs-Rest for this class only
    
    Returns:
        tuple: (fpr, tpr, roc_auc)
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
        
        fpr, tpr, _ = roc_curve(y_binary, class_probs)
        roc_auc = roc_auc_score(y_binary, class_probs)
        
        return fpr, tpr, roc_auc
    
    # Case 2: Macro-averaged ROC curve across all classes
    else:
        return _compute_macro_averaged_roc(y_test, probabilities, unique_classes)


def _compute_macro_averaged_roc(y_test, probabilities, unique_classes):
    """
    Compute macro-averaged ROC curve.
    
    Args:
        y_test: True labels
        probabilities: Probability array with shape (n_samples, n_classes)
        unique_classes: Sorted list of unique class values
    
    Returns:
        tuple: (fpr_avg, tpr_avg, roc_auc_macro)
    """
    n_classes = len(unique_classes)
    
    # Get macro-averaged ROC AUC score
    roc_auc_macro = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
    
    # Binarize the labels for One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=unique_classes)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
    
    # Compute ROC curve for each class
    fpr_per_class = []
    tpr_per_class = []
    
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
        fpr_per_class.append(fpr_i)
        tpr_per_class.append(tpr_i)
    
    # Create a common FPR grid and interpolate TPR values
    fpr_grid = np.unique(np.concatenate(fpr_per_class))
    tpr_interpolated = np.zeros_like(fpr_grid)
    
    for i in range(n_classes):
        # Interpolate TPR at common FPR points
        interp_tpr = np.interp(fpr_grid, fpr_per_class[i], tpr_per_class[i])
        tpr_interpolated += interp_tpr
    
    # Average the interpolated TPR values
    tpr_avg = tpr_interpolated / n_classes
    
    return fpr_grid, tpr_avg, roc_auc_macro


def roc(pipeline, features, model, x_test, y_test, class_index=None):
    """
    Generate ROC curve values for classification models.
    
    This function handles different classification scenarios:
    - True Binary Classification: Model naturally trained on 2 classes
    - OvR Binary Classification: Model outputs 2 probabilities (class vs rest)
    - Multi-class with class_index: One-vs-Rest ROC curve for the specified class
    - Multi-class without class_index: Macro-averaged ROC curve across all classes
    
    Args:
        pipeline: Preprocessing pipeline
        features: List of feature names
        model: Trained classification model
        x_test: Test features
        y_test: True test labels
        class_index: Optional. For multi-class models, specifies which class
                    to evaluate using One-vs-Rest approach.
    
    Returns:
        dict: Dictionary containing:
            - 'fpr': List of false positive rates
            - 'tpr': List of true positive rates
            - 'roc_auc': ROC AUC score
    
    Raises:
        ValueError: If class_index is invalid for the given data
    """
    # Preprocess the test data
    x_test_processed = preprocess(features, pipeline, x_test)
    
    # Extract probabilities from the model (all models have predict_proba)
    probabilities = model.predict_proba(x_test_processed)
    
    # Determine classification approach:
    # If model outputs 2 probabilities, treat as binary (either true binary or OvR binary)
    if probabilities.shape[1] == 2:
        # Binary pipeline handles both true binary and OvR binary cases
        fpr, tpr, roc_auc = _compute_binary_roc(y_test, probabilities)
    else:
        # Multi-class model - respect class_index parameter
        fpr, tpr, roc_auc = _compute_multiclass_roc(y_test, probabilities, class_index)
    
    # Apply decimation to reduce the number of points
    fpr_decimated, tpr_decimated = decimate_points(
        [round(num, 4) for num in list(fpr)],
        [round(num, 4) for num in list(tpr)]
    )
    
    return {
        'fpr': list(fpr_decimated),
        'tpr': list(tpr_decimated),
        'roc_auc': round(float(roc_auc), 4)
    }


def additional_roc(payload, label, folder, class_index=None):
    """
    Compute ROC curve from a payload and saved model.
    
    Args:
        payload: Dictionary containing data and metadata
        label: Target column name
        folder: Path to saved model (without .joblib extension)
        class_index: Optional class index for multi-class One-vs-Rest
    
    Returns:
        dict: ROC curve dictionary
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
    
    return roc(pipeline, payload['features'], model, x, y, class_index)
