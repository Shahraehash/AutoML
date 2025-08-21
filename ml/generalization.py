"""
Generalization of a provided model using a secondary test set.
"""

from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score,\
    confusion_matrix, classification_report, f1_score, roc_curve,\
    matthews_corrcoef
import re

from .preprocess import preprocess
from .predict import predict_ensemble
from .import_data import import_csv
from .stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci


def _extract_predictions_and_probabilities(model, x_test, threshold=0.5):
    """
    Extract predictions and probabilities from model.
    
    Args:
        model: Trained model with predict_proba method
        x_test: Preprocessed test features
        threshold: Threshold for binary classification (ignored for multi-class)
    
    Returns:
        tuple: (predictions, probabilities)
    """
    probabilities = model.predict_proba(x_test)
    
    # Binary classification with custom threshold
    if probabilities.shape[1] == 2:
        if threshold == 0.5:
            predictions = model.predict(x_test)
        else:
            # Apply custom threshold to positive class probabilities
            predictions = (probabilities[:, 1] >= threshold).astype(int)
    else:
        # Multi-class classification - threshold is ignored
        predictions = model.predict(x_test)
    
    return predictions, probabilities


def _compute_binary_generalization_report(labels, y_test, predictions, probabilities):
    """
    Compute generalization metrics for binary classification.
    
    Args:
        labels: Class labels for display
        y_test: True labels (may be multi-class for OvR scenarios)
        predictions: Predicted binary labels
        probabilities: Probability array with shape (n_samples, 2)
    
    Returns:
        dict: Generalization metrics
    """
    # Use probabilities for positive class (index 1)
    pos_probs = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
    
    # Handle OvR scenario: if y_test has more than 2 classes but we have binary probabilities,
    # we need to convert y_test to binary format for proper evaluation
    unique_classes = sorted(np.unique(y_test))
    if len(unique_classes) > 2:
        # This is an OvR scenario - convert to binary based on the most frequent class
        # or use a reasonable default conversion
        most_frequent_class = max(set(y_test), key=list(y_test).count)
        y_test_binary = (y_test == most_frequent_class).astype(int)
        predictions_binary = predictions
        
        # Adjust labels to match binary classification
        binary_labels = [f'Not {most_frequent_class}', f'{most_frequent_class}']
    else:
        # True binary classification
        y_test_binary = y_test
        predictions_binary = predictions
        binary_labels = labels
    
    # Print classification report with proper binary labels
    print('\t', classification_report(y_test_binary, predictions_binary, target_names=binary_labels, zero_division=0).replace('\n', '\n\t'))
    print('\tGeneralization:')
    
    # Calculate metrics using the binary converted data
    accuracy = accuracy_score(y_test_binary, predictions_binary)
    print('\t\tAccuracy:', accuracy)
    
    auc = roc_auc_score(y_test_binary, predictions_binary)

    roc_auc = roc_auc_score(y_test_binary, pos_probs)
    print('\t\tROC AUC:', roc_auc)
    
    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test_binary, predictions_binary).ravel()
    
    # Calculate additional metrics
    mcc = matthews_corrcoef(y_test_binary, predictions_binary)
    f1 = f1_score(y_test_binary, predictions_binary)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    prevalence = (tp + fn) / len(y_test_binary)
    
    # Get ROC curve for CI calculation
    _, tpr, _ = roc_curve(y_test_binary, pos_probs)
    
    return {
        'accuracy': round(accuracy, 4),
        'acc_95_ci': clopper_pearson(tp+tn, len(y_test_binary)),
        'mcc': round(mcc, 4),
        'avg_sn_sp': round(auc, 4),
        'roc_auc': round(roc_auc, 4),
        'roc_auc_95_ci': roc_auc_ci(roc_auc, tpr),
        'f1': round(f1, 4),
        'sensitivity': round(sensitivity, 4),
        'sn_95_ci': clopper_pearson(tp, tp+fn) if tp+fn > 0 else (0, 0),
        'specificity': round(specificity, 4),
        'sp_95_ci': clopper_pearson(tn, tn+fp) if tn+fp > 0 else (0, 0),
        'prevalence': round(prevalence, 4),
        'pr_95_ci': clopper_pearson(tp+fn, len(y_test_binary)),
        'ppv': round(tp / (tp+fp), 4) if tp+fp > 0 else 0,
        'ppv_95_ci': ppv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
        'npv': round(tn / (tn+fn), 4) if tn+fn > 0 else 0,
        'npv_95_ci': npv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
        'tn': int(tn),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp)
    }


def _compute_multiclass_generalization_report(labels, y_test, predictions, probabilities, class_index=None, threshold=0.5):
    """
    Compute generalization metrics for multi-class classification.
    
    Args:
        labels: Class labels for display
        y_test: True labels
        predictions: Predicted labels
        probabilities: Probability array with shape (n_samples, n_classes)
        class_index: If specified, compute One-vs-Rest for this class only
        threshold: Threshold for OvR binary classification when class_index is specified
    
    Returns:
        dict: Generalization metrics
    """
    unique_classes = sorted(np.unique(y_test))
    n_classes = len(unique_classes)
    
    # Validate class_index if provided
    if class_index is not None:
        if not isinstance(class_index, int) or class_index < 0 or class_index >= n_classes:
            raise ValueError(f"class_index must be an integer between 0 and {n_classes-1}")
    
    # Print classification report
    print('\t', classification_report(y_test, predictions, target_names=labels, zero_division=0).replace('\n', '\n\t'))
    print('\tGeneralization:')
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, predictions)
    print('\t\tOverall Accuracy:', accuracy)
    
    # Calculate confusion matrix for OvR decomposition
    cnf_matrix = confusion_matrix(y_test, predictions)
    
    # Case 1: Specific class One-vs-Rest
    if class_index is not None:
        actual_class_value = unique_classes[class_index]
        
        # Create binary labels for OvR evaluation
        y_binary = (y_test == actual_class_value).astype(int)
        class_probs = probabilities[:, class_index]
        
        # Apply threshold tuning for OvR binary classification
        if threshold != 0.5:
            # Use threshold-based predictions for this specific class
            pred_binary = (class_probs >= threshold).astype(int)
            print(f'\t\tUsing custom threshold {threshold} for class {actual_class_value} OvR evaluation')
        else:
            # Use original model predictions converted to binary
            pred_binary = (predictions == actual_class_value).astype(int)
        
        # Calculate confusion matrix using binary predictions
        tn, fp, fn, tp = confusion_matrix(y_binary, pred_binary).ravel()
        
        # Get class-specific ROC AUC score using probabilities
        roc_auc_class = roc_auc_score(y_binary, class_probs)
        
        # Calculate metrics for this class using threshold-tuned predictions
        accuracy_class = accuracy_score(y_binary, pred_binary)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        prevalence = (tp + fn) / len(y_test)
        
        # Class-specific F1 and MCC using threshold-tuned predictions
        f1_class = f1_score(y_binary, pred_binary)
        mcc_class = matthews_corrcoef(y_binary, pred_binary)
        
        # Get class-specific ROC curve for CI calculation
        _, tpr_curve, _ = roc_curve(y_binary, class_probs)
        
        return {
            'accuracy': round(accuracy_class, 4),
            'acc_95_ci': clopper_pearson(tp+tn, len(y_test)),
            'mcc': round(mcc_class, 4),
            'avg_sn_sp': round((sensitivity + specificity) / 2, 4),
            'roc_auc': round(roc_auc_class, 4),
            'roc_auc_95_ci': roc_auc_ci(roc_auc_class, tpr_curve),
            'f1': round(f1_class, 4),
            'sensitivity': round(sensitivity, 4),
            'sn_95_ci': clopper_pearson(tp, tp+fn) if tp+fn > 0 else (0, 0),
            'specificity': round(specificity, 4),
            'sp_95_ci': clopper_pearson(tn, tn+fp) if tn+fp > 0 else (0, 0),
            'prevalence': round(prevalence, 4),
            'pr_95_ci': clopper_pearson(tp+fn, len(y_test)),
            'ppv': round(tp / (tp+fp), 4) if tp+fp > 0 else 0,
            'ppv_95_ci': ppv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
            'npv': round(tn / (tn+fn), 4) if tn+fn > 0 else 0,
            'npv_95_ci': npv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
            'tn': int(tn),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp)
        }
    
    # Case 2: Macro-averaged metrics across all classes
    else:
        # Get macro-averaged ROC AUC score
        roc_auc_macro = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
        print('\t\tROC AUC (macro):', roc_auc_macro)
        
        # Calculate confusion matrix components for macro averaging
        fp_all = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
        fn_all = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
        tp_all = (np.diag(cnf_matrix)).astype(float)
        tn_all = (cnf_matrix.sum() - (fp_all + fn_all + tp_all)).astype(float)
        
        # Calculate per-class metrics for averaging
        all_sensitivities = []
        all_specificities = []
        all_prevalences = []
        all_f1_scores = []
        all_mcc_scores = []
        
        for class_idx in range(n_classes):
            tp = tp_all[class_idx]
            tn = tn_all[class_idx]
            fp = fp_all[class_idx]
            fn = fn_all[class_idx]
            
            # Calculate metrics for this class
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            prevalence = (tp + fn) / len(y_test)
            
            all_sensitivities.append(sensitivity)
            all_specificities.append(specificity)
            all_prevalences.append(prevalence)
            
            # Class-specific F1 and MCC for averaging
            f1_class = f1_score(y_test, predictions, labels=[class_idx], average=None)[0]
            all_f1_scores.append(f1_class)
            
            y_binary = (y_test == class_idx).astype(int)
            pred_binary = (predictions == class_idx).astype(int)
            mcc_class = matthews_corrcoef(y_binary, pred_binary)
            all_mcc_scores.append(mcc_class)
        
        # Macro-averaged metrics
        macro_sensitivity = np.mean(all_sensitivities)
        macro_specificity = np.mean(all_specificities)
        macro_prevalence = np.mean(all_prevalences)
        macro_f1 = np.mean(all_f1_scores)
        macro_mcc = np.mean(all_mcc_scores)
        
        # Sum across classes for confusion matrix totals (for CI calculations)
        tp_sum = int(np.sum(tp_all))
        tn_sum = int(np.sum(tn_all))
        fp_sum = int(np.sum(fp_all))
        fn_sum = int(np.sum(fn_all))
        
        # Get TPR for ROC AUC CI calculation
        tpr_all = tp_all / (tp_all + fn_all)
        
        return {
            'accuracy': round(accuracy, 4),
            'acc_95_ci': clopper_pearson(tp_sum+tn_sum, len(y_test)),
            'mcc': round(macro_mcc, 4),
            'avg_sn_sp': round((macro_sensitivity + macro_specificity) / 2, 4),
            'roc_auc': round(roc_auc_macro, 4),
            'roc_auc_95_ci': roc_auc_ci(roc_auc_macro, tpr_all),
            'f1': round(macro_f1, 4),
            'sensitivity': round(macro_sensitivity, 4),
            'sn_95_ci': clopper_pearson(tp_sum, tp_sum+fn_sum) if tp_sum+fn_sum > 0 else (0, 0),
            'specificity': round(macro_specificity, 4),
            'sp_95_ci': clopper_pearson(tn_sum, tn_sum+fp_sum) if tn_sum+fp_sum > 0 else (0, 0),
            'prevalence': round(macro_prevalence, 4),
            'pr_95_ci': clopper_pearson(tp_sum+fn_sum, len(y_test)),
            'ppv': round(tp_sum / (tp_sum+fp_sum), 4) if tp_sum+fp_sum > 0 else 0,
            'ppv_95_ci': ppv_95_ci(macro_sensitivity, macro_specificity, tp_sum+fn_sum, fp_sum+tn_sum, macro_prevalence),
            'npv': round(tn_sum / (tn_sum+fn_sum), 4) if tn_sum+fn_sum > 0 else 0,
            'npv_95_ci': npv_95_ci(macro_sensitivity, macro_specificity, tp_sum+fn_sum, fp_sum+tn_sum, macro_prevalence),
            'tn': tn_sum,
            'tp': tp_sum,
            'fn': fn_sum,
            'fp': fp_sum
        }


def _generate_labels(y_test, n_prob_classes, provided_labels=None):
    unique_classes = sorted(np.unique(y_test))
    n_classes = len(unique_classes)
    
    # If labels are provided and match the number of classes, use them
    if provided_labels is not None and len(provided_labels) == n_classes:
        return provided_labels
    
    # Generate appropriate labels based on classification type
    if n_classes == 2:
        return ['Class 0', 'Class 1']
    else:
        return [f'Class {int(cls)}' for cls in unique_classes]


def generalize(pipeline, features, model, x_test, y_test, labels=None, threshold=0.5, class_index=None):
    """
    Compute generalization metrics for a trained model.
    
    This function handles different classification scenarios:
    - True Binary Classification: Model naturally trained on 2 classes
    - OvR Binary Classification: Model outputs 2 probabilities (class vs rest)
    - Multi-class with class_index: One-vs-Rest evaluation for the specified class
    - Multi-class without class_index: Macro-averaged metrics across all classes
    
    Args:
        pipeline: Preprocessing pipeline
        features: List of feature names
        model: Trained classification model
        x_test: Test features
        y_test: True test labels
        labels: Optional class labels for display
        threshold: Threshold for binary classification (default 0.5)
        class_index: Optional. For multi-class models, specifies which class
                    to evaluate using One-vs-Rest approach.
    
    Returns:
        dict: Generalization metrics dictionary
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Preprocess the test data
    x_test_processed = preprocess(features, pipeline, x_test)
    
    # Extract predictions and probabilities
    predictions, probabilities = _extract_predictions_and_probabilities(model, x_test_processed, threshold)
    
    # Generate appropriate labels
    labels = _generate_labels(y_test, probabilities.shape[1], labels)
    
    # Determine classification approach:
    # If model outputs 2 probabilities, treat as binary (either true binary or OvR binary)
    if probabilities.shape[1] == 2:
        # Binary pipeline handles both true binary and OvR binary cases
        return _compute_binary_generalization_report(labels, y_test, predictions, probabilities)
    else:
        # Multi-class model - respect class_index parameter
        return _compute_multiclass_generalization_report(labels, y_test, predictions, probabilities, class_index)


def generalize_model(payload, label, folder, threshold=0.5, class_index=None):
    """
    Compute generalization metrics from a payload and saved model.
    
    Args:
        payload: Dictionary containing data and metadata
        label: Target column name
        folder: Path to saved model (without .joblib extension)
        threshold: Threshold for binary classification (default 0.5)
        class_index: Optional class index for multi-class One-vs-Rest
    
    Returns:
        dict: Generalization metrics dictionary
    """
    # Load and prepare data
    initial_df = pd.DataFrame(payload['data'], columns=payload['columns'])
    
    # Apply numeric conversion and drop NaN rows
    numeric_df = initial_df.apply(pd.to_numeric, errors='coerce')
    data = numeric_df.dropna()
    
    if data.empty:
        raise ValueError("No valid data remaining after preprocessing")
    
    # Extract features and labels
    x = data[payload['features']].to_numpy()
    y = data[label].to_numpy()
    
    # Load the trained pipeline
    pipeline = load(folder + '.joblib')
    
    # Extract the model from the pipeline
    model = pipeline.steps[-1][1]
    
    # Preprocess the test data using the same pipeline transformations
    x_processed = preprocess(payload['features'], pipeline, x)
    
    # Extract predictions and probabilities from preprocessed data
    predictions, probabilities = _extract_predictions_and_probabilities(model, x_processed, threshold)
    
    # Generate appropriate labels based on classification type
    if probabilities.shape[1] == 2:
        labels = ['No ' + label, label]
    else:
        unique_classes = sorted(np.unique(y))
        labels = [f'Class {int(cls)}' for cls in unique_classes]
    
    # Determine classification approach:
    # If model outputs 2 probabilities, treat as binary (either true binary or OvR binary)
    if probabilities.shape[1] == 2:
        # Binary pipeline handles both true binary and OvR binary cases
        result = _compute_binary_generalization_report(labels, y, predictions, probabilities)
    else:
        # Multi-class model - respect class_index parameter and threshold for OvR tuning
        result = _compute_multiclass_generalization_report(labels, y, predictions, probabilities, class_index, threshold)
    
    # Add metadata for proper frontend calculation
    unique_classes = sorted(np.unique(y))
    num_classes = len(unique_classes)
    is_multiclass = num_classes > 2
    actual_processed_rows = len(y)
    
    # Add the metadata to the result
    result['num_classes'] = num_classes
    result['is_multiclass'] = is_multiclass
    result['actual_processed_rows'] = actual_processed_rows
    
    return result


def generalize_ensemble(total_models, job_folder, dataset_folder, label):
    """
    Compute generalization metrics for ensemble models.
    
    Args:
        total_models: Number of models in ensemble
        job_folder: Path to job folder containing models
        dataset_folder: Path to dataset folder
        label: Target column name
    
    Returns:
        dict: Dictionary containing soft and hard generalization results
    """
    # Load test data
    x_test, y_test, feature_names, _, _, label_mapping_info = import_csv(dataset_folder + '/test.csv', label)
    
    data = pd.DataFrame(x_test, columns=feature_names)
    
    # Get ensemble predictions
    soft_result = predict_ensemble(total_models, data, job_folder, 'soft')
    hard_result = predict_ensemble(total_models, data, job_folder, 'hard')
    
    # Generate appropriate labels
    if label_mapping_info and 'original_labels' in label_mapping_info:
        original_labels = label_mapping_info['original_labels']
        if len(original_labels) == 2:
            labels = ['No ' + label, label]
        else:
            labels = [f'Class {int(cls)}' for cls in original_labels]
    else:
        # Fallback to current logic for backward compatibility
        unique_labels = sorted(y_test.unique())
        if len(unique_labels) == 2:
            labels = ['No ' + label, label]
        else:
            labels = [f'Class {int(cls)}' for cls in unique_labels]
    
    # Determine classification type for both results
    soft_probs = soft_result['probability']
    hard_probs = hard_result['probability']
    
    # Process soft voting results
    # If model outputs 2 probabilities, treat as binary (either true binary or OvR binary)
    if soft_probs.shape[1] == 2:
        soft_gen = _compute_binary_generalization_report(labels, y_test, soft_result['predicted'], soft_probs)
    else:
        soft_gen = _compute_multiclass_generalization_report(labels, y_test, soft_result['predicted'], soft_probs)
    
    # Process hard voting results
    # If model outputs 2 probabilities, treat as binary (either true binary or OvR binary)
    if hard_probs.shape[1] == 2:
        hard_gen = _compute_binary_generalization_report(labels, y_test, hard_result['predicted'], hard_probs)
    else:
        hard_gen = _compute_multiclass_generalization_report(labels, y_test, hard_result['predicted'], hard_probs)
    
    return {
        'soft_generalization': soft_gen,
        'hard_generalization': hard_gen
    }


# Legacy function for backward compatibility
def generalization_report(labels, y_test, predictions, probabilities, class_index=None):
    """
    Legacy function for backward compatibility.
    
    This function is deprecated. Use the main generalize() function instead.
    """
    # Generate appropriate labels if not provided
    labels = _generate_labels(y_test, probabilities.shape[1], labels)
    
    # Determine classification approach:
    # If model outputs 2 probabilities, treat as binary (either true binary or OvR binary)
    if probabilities.shape[1] == 2:
        # Binary pipeline handles both true binary and OvR binary cases
        return _compute_binary_generalization_report(labels, y_test, predictions, probabilities)
    else:
        # Multi-class model - respect class_index parameter
        return _compute_multiclass_generalization_report(labels, y_test, predictions, probabilities, class_index)
