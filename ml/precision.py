"""
Compute precision recall curve and precision score
"""

import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import precision_score, recall_score, precision_recall_curve

from .preprocess import preprocess
from .utils import decimate_points


def _compute_binary_precision_recall(y_test, probabilities):
    """
    Compute precision-recall curve for binary classification.
    
    Args:
        y_test: True labels (may be multi-class for OvR scenarios)
        probabilities: Probability array with shape (n_samples, 2)
    
    Returns:
        tuple: (precision, recall)
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
    
    # Compute precision-recall curve using binary labels
    precision, recall, _ = precision_recall_curve(y_test_binary, pos_probs)
    
    return precision, recall


def _compute_multiclass_precision_recall(y_test, probabilities, class_index=None):
    """
    Compute precision-recall curve for multi-class classification.
    
    Args:
        y_test: True labels
        probabilities: Probability array with shape (n_samples, n_classes)
        class_index: If specified, compute One-vs-Rest for this class only
    
    Returns:
        tuple: (precision, recall)
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
        
        precision, recall, _ = precision_recall_curve(y_binary, class_probs)
        return precision, recall
    
    # Case 2: Macro-averaged curve across all classes
    else:
        return compute_macro_averaged_curve(y_test, probabilities, unique_classes)


def compute_macro_averaged_curve(y_test, probabilities, unique_classes,use_proba=True):
    """
    Compute macro-averaged precision-recall curve.
    
    Args:
        y_test: True labels
        probabilities: Probability array with shape (n_samples, n_classes)
        unique_classes: Sorted list of unique class values
    
    Returns:
        tuple: (precision_avg, recall_common)
    """
    # Use common recall points for interpolation
    common_recall = np.linspace(0, 1, 101)  # 101 points from 0 to 1
    precision_interp_curves = []
    
    for class_idx, class_val in enumerate(unique_classes):
        # Create binary labels for current class vs rest
        y_binary = (y_test == class_val).astype(int)
        
        # Skip if no positive samples for this class
        if y_binary.sum() == 0:
            continue
            
        # Get probabilities for this class
        class_probs = probabilities[:, class_idx]
        
        # Compute precision-recall curve
        prec_class, rec_class, _ = precision_recall_curve(y_binary, class_probs)
        
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


def precision_recall(pipeline, features, model, x_test, y_test, class_index=None):
    """
    Compute precision-recall curve for classification models.
    
    This function handles different classification scenarios:
    - True Binary Classification: Model naturally trained on 2 classes
    - OvR Binary Classification: Model outputs 2 probabilities (class vs rest)
    - Multi-class with class_index: One-vs-Rest curve for the specified class
    - Multi-class without class_index: Macro-averaged curve across all classes
    
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
            - 'precision': List of precision values
            - 'recall': List of recall values
    
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
        precision, recall = _compute_binary_precision_recall(y_test, probabilities)
    else:
        # Multi-class model - respect class_index parameter
        precision, recall = _compute_multiclass_precision_recall(y_test, probabilities, class_index)
    
    # Apply decimation to reduce the number of points
    recall_decimated, precision_decimated = decimate_points(
        [round(num, 4) for num in list(recall)],
        [round(num, 4) for num in list(precision)]
    )
    
    return {
        'precision': list(precision_decimated),
        'recall': list(recall_decimated)
    }


def additional_precision(payload, label, folder, class_index=None):
    """
    Compute precision-recall curve from a payload and saved model.
    
    Args:
        payload: Dictionary containing data and metadata
        label: Target column name
        folder: Path to saved model (without .joblib extension)
        class_index: Optional class index for multi-class One-vs-Rest
    
    Returns:
        dict: Precision-recall curve dictionary
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
    
    return precision_recall(pipeline, payload['features'], model, x, y, class_index)
