"""
Multiclass Macro-Averaging Classifier

This module provides the MulticlassMacroClassifier class for handling multiclass classification
tasks with macro-averaged metrics. It extends the AutoMLClassifier base class with 
multiclass-specific logic.
"""

import json
import numpy as np
import pandas as pd
import os
from timeit import default_timer as timer
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.preprocessing import label_binarize
from nyoka import skl_to_pmml, xgboost_to_pmml


from .base_classifier import AutoMLClassifier
from ..preprocess import preprocess
from ..utils import model_key_to_name, decimate_points, explode_key
from ..stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci
from ..processors.estimators import ESTIMATORS, get_xgb_classifier
from ..processors.scorers import SCORER_NAMES
from ..processors.scalers import SCALERS
from ..processors.feature_selection import FEATURE_SELECTORS
from ..import_data import import_data


class MulticlassMacroClassifier(AutoMLClassifier):
    """
    Handles multiclass classification tasks with macro-averaged metrics.
    
    This classifier is optimized for datasets with more than 2 classes and provides
    macro-averaged evaluation metrics across all classes.
    """
    
    def __init__(self, parameters, output_path='.', update_function=lambda x, y: None):
        """
        Initialize the Multiclass Macro Classifier.
        
        Args:
            parameters (dict): Configuration parameters for the classifier
            output_path (str): Path where results and models will be saved
            update_function (callable): Callback function for progress updates
        """
        super().__init__(parameters, output_path, update_function)
        self.ovr_models = {}
    
    def evaluate_precision_recall(self, pipeline, features, estimator, x_test, y_test, class_index=None):
        """
        Evaluate precision-recall for multiclass classification with macro-averaging.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            class_index (int, optional): Specific class index for OvR evaluation
            
        Returns:
            dict: Precision-recall evaluation results with macro-averaging
        """
        x_test = preprocess(features, pipeline, x_test)
        probabilities = estimator.predict_proba(x_test)
        
        if class_index is not None:
            # OvR single class evaluation - binary precision-recall for this class vs rest
            y_binary = (y_test == class_index).astype(int)
            class_probabilities = probabilities[:, class_index]
            precision, recall, _ = precision_recall_curve(y_binary, class_probabilities)
            
            # Apply decimation
            recall, precision = decimate_points(
                [round(num, 4) for num in list(recall)],
                [round(num, 4) for num in list(precision)]
            )
            
            return {
                'precision': list(precision),
                'recall': list(recall)
            }
        else:
            # Macro-averaged evaluation across all classes
            unique_classes = sorted(np.unique(y_test))
            precision_avg, recall_avg = self._compute_macro_averaged_curve(y_test, probabilities, unique_classes)
            
            return {
                'precision': [round(num, 4) for num in list(precision_avg)],
                'recall': [round(num, 4) for num in list(recall_avg)]
            }
    
    def _compute_macro_averaged_curve(self, y_test, scores_or_probs, unique_classes, use_proba=True):
        """Helper function to compute macro-averaged precision-recall curve"""
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
    
    def evaluate_reliability(self, pipeline, features, estimator, x_test, y_test, class_index=None):
        """
        Evaluate reliability for multiclass classification with macro-averaging.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            class_index (int, optional): Specific class index for OvR evaluation
            
        Returns:
            dict: Reliability evaluation results with macro-averaging
        """
        x_test = preprocess(features, pipeline, x_test)
        
        n_classes = len(np.unique(y_test))

        if hasattr(estimator, 'decision_function'):
            probabilities = estimator.decision_function(x_test)

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
                probabilities = estimator.predict_proba(x_test)[:, 1]
                fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')
                brier_score = brier_score_loss(y_test, probabilities)
                
            # Multi-class Classification
            else:
                probabilities = estimator.predict_proba(x_test)
            
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
    
    def evaluate_roc(self, pipeline, features, estimator, x_data, y_data, class_index=None):
        """
        Evaluate ROC for multiclass classification with macro-averaging.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_data (array): Features for ROC evaluation
            y_data (array): Labels for ROC evaluation
            class_index (int, optional): Specific class index for OvR evaluation
            
        Returns:
            dict: ROC evaluation results with macro-averaging
        """
        x_data = preprocess(features, pipeline, x_data)
        
        probabilities = estimator.predict_proba(x_data)
        predictions = estimator.predict(x_data)
        
        # Check if this is binary or multiclass classification
        n_classes = probabilities.shape[1]
        
        # Binary classification
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_data, probabilities[:, 1])
            roc_auc = roc_auc_score(y_data, probabilities[:, 1])

        # Multiclass classification
        else:
            # If class_index is specified, return One-vs-Rest curve for that class
            if class_index is not None:
                unique_classes = sorted(np.unique(y_data))
                if class_index < len(unique_classes):
                    actual_class_value = unique_classes[class_index]
                    y_binary = (y_data == actual_class_value).astype(int)
                    fpr, tpr, _ = roc_curve(y_binary, probabilities[:, class_index])
                    roc_auc = roc_auc_score(y_binary, probabilities[:, class_index])
                else:
                    # Invalid class index, fall back to macro average
                    class_index = None
            
            # If class_index is None or invalid, return macro-averaged curve (original behavior)
            if class_index is None:
                roc_auc = roc_auc_score(y_data, probabilities, multi_class='ovr', average='macro') 
                
                y_test_bin = label_binarize(y_data, classes=np.unique(y_data))
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
    
    def predict(self, data, model_key, class_index=None):
        """
        Make predictions using a specific trained multiclass model.
        
        Args:
            data: Input data for prediction (numpy array or similar)
            model_key (str): Key identifying the model
            class_index (int, optional): Specific class index for OvR-style prediction
            
        Returns:
            dict: Prediction results with multiclass-specific logic
        """
        if model_key not in self.main_models:
            raise KeyError(f"Model {model_key} not found")
        
        model = self.main_models[model_key]
        
        # Multiclass classification prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
            predictions = model.predict(data)
            
            if class_index is not None:
                # Return class-specific probabilities
                class_probabilities = probabilities[:, class_index]
                return {
                    'predicted': predictions.tolist(),
                    'probability': class_probabilities.tolist(),
                    'all_probabilities': probabilities.tolist(),
                    'class_index': class_index,
                    'classification_type': 'multiclass_macro'
                }
            else:
                # Return max probabilities (confidence in prediction)
                max_probabilities = probabilities.max(axis=1)
                return {
                    'predicted': predictions.tolist(),
                    'probability': max_probabilities.tolist(),
                    'all_probabilities': probabilities.tolist(),
                    'classification_type': 'multiclass_macro'
                }
        else:
            # Fallback for models without predict_proba
            predictions = model.predict(data)
            return {
                'predicted': predictions.tolist(),
                'probability': [1.0] * len(predictions),  # Default probability
                'classification_type': 'multiclass_macro'
            }
    
    def evaluate_generalization(self, pipeline, features, estimator, x_test, y_test, labels):
        """
        Evaluate generalization for multiclass classification with macro-averaging.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            labels (list): Class labels
            
        Returns:
            dict: Generalization evaluation results with macro-averaging
        """
        # Process test data based on pipeline
        x_test = preprocess(features, pipeline, x_test)
        proba = estimator.predict_proba(x_test)
        predictions = estimator.predict(x_test)
        probabilities = proba
        
        return self.generalization_report(labels, y_test, predictions, probabilities)
    
    def generalization_report(self, labels, y2, predictions, probabilities, class_index=None):
        """
        Generate generalization report for multiclass classification with macro-averaging.
        
        Args:
            labels (list): Class labels
            y2 (array): True labels
            predictions (array): Predicted labels
            probabilities (array): Prediction probabilities
            class_index (int, optional): Specific class index for OvR evaluation
            
        Returns:
            dict: Multiclass generalization metrics with macro-averaging
        """
        # Get unique classes from actual data
        unique_classes = sorted(np.unique(y2))
        n_classes = len(unique_classes)
        
        # If labels are provided but don't match the number of classes, generate appropriate labels
        if labels is None or len(labels) != n_classes:
            labels = [f'Class {int(cls)}' for cls in unique_classes]
        
        print('\t', classification_report(y2, predictions, target_names=labels).replace('\n', '\n\t'))
        print('\tGeneralization:')
        
        # Check if we need class-specific calculations for one class only
        if class_index is not None:
            # Single class metrics
            
            # Get class-specific ROC AUC score
            roc_auc_scores = roc_auc_score(y2, probabilities, multi_class='ovr', average=None)
            roc_auc_class = roc_auc_scores[class_index]
            
            # Calculate confusion matrix for OvR decomposition
            cnf_matrix = confusion_matrix(y2, predictions)
            
            # For OvR: current class vs all others
            tp = cnf_matrix[class_index, class_index]
            fn = np.sum(cnf_matrix[class_index, :]) - tp
            fp = np.sum(cnf_matrix[:, class_index]) - tp
            tn = np.sum(cnf_matrix) - tp - fn - fp
            
            # Calculate metrics for this class
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            prevalence = (tp + fn) / len(y2)
            
            # Class-specific F1 and MCC
            f1_class = f1_score(y2, predictions, labels=[class_index], average=None)[0]
            y_binary = (y2 == class_index).astype(int)
            pred_binary = (predictions == class_index).astype(int)
            mcc_class = matthews_corrcoef(y_binary, pred_binary)
            
            # Get class-specific ROC curve for CI calculation
            y_binary_prob = probabilities[:, class_index] if probabilities.ndim > 1 else probabilities
            fpr, tpr_curve, _ = roc_curve(y_binary, y_binary_prob)
            
            return {
                'accuracy': round(accuracy_score(y2, predictions), 4),
                'acc_95_ci': clopper_pearson(tp+tn, len(y2)),
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
                'pr_95_ci': clopper_pearson(tp+fn, len(y2)),
                'ppv': round(tp / (tp+fp), 4) if tp+fp > 0 else 0,
                'ppv_95_ci': ppv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
                'npv': round(tn / (tn+fn), 4) if tn+fn > 0 else 0,
                'npv_95_ci': npv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
                'tn': int(tn),
                'tp': int(tp),
                'fn': int(fn),
                'fp': int(fp)
            }
        
        else:
            # Macro-averaged metrics across all classes
            # Overall accuracy (same for all classes)
            accuracy = accuracy_score(y2, predictions)
            print('\t\tOverall Accuracy:', accuracy)
            
            # Calculate confusion matrix for OvR decomposition
            cnf_matrix = confusion_matrix(y2, predictions)
            
            # Get macro-averaged ROC AUC score
            roc_auc_macro = roc_auc_score(y2, probabilities, multi_class='ovr', average='macro')
            print('\t\tROC AUC (macro):', roc_auc_macro)
            
            # Calculate per-class metrics for averaging
            all_sensitivities = []
            all_specificities = []
            all_prevalences = []
            all_f1_scores = []
            all_mcc_scores = []
            
            # Calculate confusion matrix components for macro averaging
            fp_all = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
            fn_all = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
            tp_all = (np.diag(cnf_matrix)).astype(float)
            tn_all = (cnf_matrix.sum() - (fp_all + fn_all + tp_all)).astype(float)
            
            # Get TPR for ROC AUC CI calculation
            tpr_all = tp_all / (tp_all + fn_all)
            
            for class_idx in range(n_classes):
                tp = tp_all[class_idx]
                tn = tn_all[class_idx]
                fp = fp_all[class_idx]
                fn = fn_all[class_idx]
                
                # Calculate metrics for this class
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                prevalence = (tp + fn) / len(y2)
                
                all_sensitivities.append(sensitivity)
                all_specificities.append(specificity)
                all_prevalences.append(prevalence)
                
                # Class-specific F1 and MCC for averaging
                f1_class = f1_score(y2, predictions, labels=[class_idx], average=None)[0]
                all_f1_scores.append(f1_class)
                
                y_binary = (y2 == class_idx).astype(int)
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
            
            return {
                'accuracy': round(accuracy, 4),
                'acc_95_ci': clopper_pearson(tp_sum+tn_sum, len(y2)),
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
                'pr_95_ci': clopper_pearson(tp_sum+fn_sum, len(y2)),
                'ppv': round(tp_sum / (tp_sum+fp_sum), 4) if tp_sum+fp_sum > 0 else 0,
                'ppv_95_ci': ppv_95_ci(macro_sensitivity, macro_specificity, tp_sum+fn_sum, fp_sum+tn_sum, macro_prevalence),
                'npv': round(tn_sum / (tn_sum+fn_sum), 4) if tn_sum+fn_sum > 0 else 0,
                'npv_95_ci': npv_95_ci(macro_sensitivity, macro_specificity, tp_sum+fn_sum, fp_sum+tn_sum, macro_prevalence),
                'tn': tn_sum,
                'tp': tp_sum,
                'fn': fn_sum,
                'fp': fp_sum
            }
    
    
    def _compute_binary_class_results(self, pipeline, features, estimator, x_val, y_val, x_train=None, y_train=None, x_test=None, y_test=None):
        """Compute metrics for binary classification (used for OvR models in re-optimization mode)"""
        
        # Compute reliability, precision_recall, and roc for binary classification using class methods
        generalization_data = self.generalization_report(None, y_val, estimator.predict(preprocess(features, pipeline, x_val)), estimator.predict_proba(preprocess(features, pipeline, x_val)))
        reliability_data = self.evaluate_reliability(pipeline, features, estimator, x_val, y_val)
        precision_data = self.evaluate_precision_recall(pipeline, features, estimator, x_val, y_val)
        roc_data = self.evaluate_roc(pipeline, features, estimator, x_val, y_val)
        
        # Compute training ROC AUC if training data is provided
        training_roc_auc = None
        roc_delta = None
        
        if x_train is not None and y_train is not None:
            training_roc_data = self.evaluate_roc(pipeline, features, estimator, x_train, y_train)
            training_roc_auc = training_roc_data['roc_auc']
            
            # Calculate ROC delta
            if roc_data['roc_auc'] is not None and training_roc_auc is not None:
                roc_delta = round(abs(roc_data['roc_auc'] - training_roc_auc), 4)
        
        # Compute test ROC metrics if test data is provided
        test_roc_data = None
        if x_test is not None and y_test is not None:
            test_roc_data = self.evaluate_roc(pipeline, features, estimator, x_test, y_test)
        
        return {
            'generalization': generalization_data,
            'reliability': reliability_data,
            'precision_recall': precision_data,
            'roc_auc': roc_data,
            'training_roc_auc': training_roc_auc,
            'roc_delta': roc_delta,
            'test_roc_data': test_roc_data
        }

    def _compute_class_specific_results(self, pipeline, features, estimator, x_val, y_val, n_classes, model_key, class_idx, x_train=None, y_train=None, x_test_orig=None, y_test_orig=None):
        """Compute class-specific results for multiclass classification"""
        
        # Compute class-specific metrics using class methods with class_index parameter
        x_val_processed = preprocess(features, pipeline, x_val)
        predictions = estimator.predict(x_val_processed)
        probabilities = estimator.predict_proba(x_val_processed)
        
        generalization_data = self.generalization_report(None, y_val, predictions, probabilities, class_index=class_idx)
        reliability_data = self.evaluate_reliability(pipeline, features, estimator, x_val, y_val, class_index=class_idx)
        precision_data = self.evaluate_precision_recall(pipeline, features, estimator, x_val, y_val, class_index=class_idx)
        roc_data = self.evaluate_roc(pipeline, features, estimator, x_val, y_val, class_index=class_idx)
        
        # Compute training ROC AUC for this class if training data is provided
        training_roc_auc = None
        roc_delta = None
        
        if x_train is not None and y_train is not None:
            training_roc_data = self.evaluate_roc(pipeline, features, estimator, x_train, y_train, class_index=class_idx)
            training_roc_auc = training_roc_data['roc_auc']
            
            # Calculate ROC delta (absolute difference between generalization and training)
            if roc_data['roc_auc'] is not None and training_roc_auc is not None:
                roc_delta = round(abs(roc_data['roc_auc'] - training_roc_auc), 4)
        
        # Compute test ROC metrics if original test data is provided
        test_roc_data = None
        if x_test_orig is not None and y_test_orig is not None:
            test_roc_data = self.evaluate_roc(pipeline, features, estimator, x_test_orig, y_test_orig, class_index=class_idx)
        
        return {
            'generalization': generalization_data,
            'reliability': reliability_data,
            'precision_recall': precision_data,
            'roc_auc': roc_data,
            'training_roc_auc': training_roc_auc,
            'roc_delta': roc_delta,
            'test_roc_data': test_roc_data
        }

    def _create_ovr_csv_entry(self, main_result, class_idx, ovr_best_params, class_metrics):
        """Create a CSV entry for an OvR model using already computed metrics"""

        # Create base CSV entry
        ovr_result = {
            'key': f"{main_result['key']}_ovr_class_{class_idx}",
            'class_type': 'ovr',
            'class_index': class_idx,
            'scaler': main_result['scaler'],
            'feature_selector': main_result['feature_selector'],
            'algorithm': main_result['algorithm'],
            'searcher': main_result['searcher'],
            'scorer': main_result['scorer']
        }
        
        ovr_result.update(class_metrics.get('generalization'))
        
        # Add basic metadata
        ovr_result.update({
            'selected_features': main_result['selected_features'],
            'feature_scores': main_result['feature_scores'],
            'best_params': ovr_best_params
        })
        
        test_data = class_metrics['test_roc_data']
        ovr_result['test_fpr'] = test_data.get('fpr')
        ovr_result['test_tpr'] = test_data.get('tpr')

        ovr_result['training_roc_auc'] = class_metrics.get('training_roc_auc')
        ovr_result['roc_delta'] = class_metrics.get('roc_delta')

        roc_data = class_metrics['roc_auc']
        ovr_result['generalization_fpr'] = roc_data.get('fpr')
        ovr_result['generalization_tpr'] = roc_data.get('tpr')
        
        reliability_data = class_metrics['reliability']
        ovr_result['brier_score'] = reliability_data.get('brier_score')
        ovr_result['fop'] = reliability_data.get('fop')
        ovr_result['mpv'] = reliability_data.get('mpv')

        pr_data = class_metrics['precision_recall']
        ovr_result['precision'] = pr_data.get('precision')
        ovr_result['recall'] = pr_data.get('recall')
        
        return ovr_result

    def _generate_ovr_models_and_results(self, pipeline, features, main_model, main_result, 
                                        x_train, y_train, x_test, y_test, x2, y2, labels,
                                        estimator, scorer, reoptimize_ovr=False):
        """Generate OvR models and return both CSV entries and class data for .pkl.gz storage"""
        
        n_classes = len(np.unique(y_train))
        unique_classes = sorted(np.unique(y_train))
        
        csv_entries = []
        ovr_models = {}
        total_fits = 0
        
        # Storage for class-specific data (for .pkl.gz file)
        all_class_data = {
            'model_key': main_result['key'],
            'n_classes': n_classes,
            'class_data': {}
        }
        
        for class_idx in range(n_classes):
            # Create binary labels for this class vs rest
            actual_class_value = unique_classes[class_idx]
            y_binary = (y_train == actual_class_value).astype(int)
            y_test_binary = (y_test == actual_class_value).astype(int)
            y2_binary = (y2 == actual_class_value).astype(int)
            
            if reoptimize_ovr:
                # Re-optimization mode: Train actual OvR model
                ovr_candidates = self.refit_candidates(
                    pipeline, features, estimator, scorer, x_train, y_binary
                )
                
                ovr_model = ovr_candidates[0]['best_estimator']
                ovr_best_params = ovr_candidates[0]['best_params']
                total_fits += len(ovr_candidates)
                
                # Store OvR model
                ovr_key = f"{main_result['key']}_ovr_class_{class_idx}"
                ovr_models[ovr_key] = ovr_model

                # Re-optimization mode: Use binary classification path
                # OvR model was trained on binary data, so evaluate it on binary data
                class_metrics = self._compute_binary_class_results(
                    pipeline, features, ovr_model, 
                    x2, y2_binary,           # Binary generalization data
                    x_train, y_binary,       # Binary training data
                    x_test, y_test_binary    # Binary test data
                )
                
            else:
                # Efficient mode: Use main model
                ovr_model = main_model
                ovr_best_params = main_result['best_params']
            
                # Efficient mode: Use multiclass classification path with class_idx
                class_metrics = self._compute_class_specific_results(
                    pipeline, features, ovr_model, x2, y2, n_classes, 
                    main_result['key'], class_idx, x_train, y_train, x_test, y_test
                )
            
            # Store class data for .pkl.gz file
            all_class_data['class_data'][class_idx] = class_metrics
            
            # Create CSV entry for this OvR model using already computed metrics
            csv_entry = self._create_ovr_csv_entry(
                main_result, class_idx, ovr_best_params, class_metrics
            )
            
            csv_entries.append(csv_entry)
        
        return csv_entries, all_class_data, ovr_models, total_fits

    def create_model(self, key, hyper_parameters, selected_features, dataset_path=None, label_column=None, output_path='.', threshold=.5):
        """
        Create and export a multiclass classification model (adapted from outdated/create_model.py)
        
        Args:
            key (str): Model key identifying the pipeline configuration
            hyper_parameters (dict): Hyperparameters for the model
            selected_features (list): List of selected feature names
            dataset_path (str): Path to dataset folder containing train.csv and test.csv
            label_column (str): Name of the target column
            output_path (str): Path where outputs will be saved
            threshold (float): Not used for multiclass classification
            
        Returns:
            dict: Generalization results for the created model
        """
        if dataset_path is None:
            print('Missing dataset path')
            return {}

        if label_column is None:
            print('Missing column name for classifier target')
            return {}

        # Import data
        (x_train, _, y_train, _, x2, y2, features, _) = \
            import_data(dataset_path + '/train.csv', dataset_path + '/test.csv', label_column)

        # Validate multiclass classification
        n_classes = len(np.unique(y_train))
        if n_classes <= 2:
            raise ValueError(f"MulticlassMacroClassifier.create_model expects more than 2 classes, got {n_classes}")

        # Get pipeline details from the key
        scaler, feature_selector, estimator, _, _ = explode_key(key)
        steps = []

        # Drop the unused features
        if 'pca-' not in feature_selector:
            for index, feature in reversed(list(enumerate(features))):
                if feature not in selected_features:
                    x_train = np.delete(x_train, index, axis=1)
                    x2 = np.delete(x2, index, axis=1)

        # Add the scaler, if used
        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        # Add the feature transformer
        if 'pca-' in feature_selector:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        # Add the estimator with proper XGBoost configuration
        if estimator == 'gb':
            base_estimator = get_xgb_classifier(n_classes)
        else:
            base_estimator = ESTIMATORS[estimator]
        
        steps.append(('estimator', base_estimator.set_params(**hyper_parameters)))

        # Fit the pipeline using the same training data
        pipeline = Pipeline(steps)
        model = self.train_model(pipeline, selected_features, x_train, y_train)

        # If the model is DNN or RF, attempt to swap the estimator for a pickled one
        if os.path.exists(output_path + '/models/' + key + '.joblib'):
            pickled_estimator = load(output_path + '/models/' + key + '.joblib')
            pipeline = Pipeline(pipeline.steps[:-1] + [('estimator', pickled_estimator)])

        # Multiclass classification labels
        unique_labels = sorted(y2.unique())
        labels = [f'Class {int(cls)}' for cls in unique_labels]

        # Assess the model performance and store the results
        generalization_result = self.evaluate_generalization(pipeline, model['features'], pipeline['estimator'], x2, y2, labels)
        with open(output_path + '/pipeline.json', 'w') as statsfile:
            json.dump(generalization_result, statsfile)

        # Dump the pipeline to a file
        dump(pipeline, output_path + '/pipeline.joblib')
        pd.DataFrame([selected_features]).to_csv(output_path + '/input.csv', index=False, header=False)

        # Export the model as a PMML
        try:
            if estimator == 'gb':
                if xgboost_to_pmml:
                    xgboost_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
            else:
                if skl_to_pmml:
                    skl_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
        except Exception:
            try:
                os.remove(output_path + '/pipeline.pmml')
            except OSError:
                pass

        return generalization_result

    def fit(self, x_train, x_val, y_train, y_val, x_test, y_test, feature_names, labels):
        """
        Train multiclass classification models with macro-averaging.
        
        Args:
            x_train (array): Training features (for model fitting)
            x_val (array): Validation features (for hyperparameter tuning)
            y_train (array): Training labels (for model fitting)
            y_val (array): Validation labels (for hyperparameter tuning)
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            feature_names (list): List of feature names
            labels (list): List of class labels
            
        Returns:
            bool: True if successful, False otherwise
        """
        start = timer()
        n_classes = len(np.unique(y_train))
        
        # Validate that this is indeed multiclass classification
        if n_classes <= 2:
            raise ValueError(f"MulticlassMacroClassifier expects more than 2 classes, got {n_classes}")
        
        # Initialize reports
        self.initialize_reports()
        
        # Generate all pipeline combinations
        all_pipelines = self.generate_pipeline_combinations()
        
        print(f"Starting multiclass macro-averaging classification with {len(all_pipelines)} pipeline combinations...")
        print(f"Number of classes: {n_classes}")
        
        for index, (estimator, scaler, feature_selector, searcher) in enumerate(all_pipelines):
            
            # Trigger callback for task monitoring
            self.update_function(index, len(all_pipelines))
            
            key = '__'.join([scaler, feature_selector, estimator, searcher])
            print('Generating ' + model_key_to_name(key))
            
            # Generate the pipeline
            pipeline_result = self.create_pipeline(estimator, scaler, feature_selector, searcher, y_train)
            pipeline = pipeline_result[0]
            pipeline_fits = pipeline_result[1]
            
            # Track total fits
            if estimator not in self.total_fits:
                self.total_fits[estimator] = 0
            self.total_fits[estimator] += pipeline_fits
            
            # Fit the pipeline
            model = self.train_model(pipeline, feature_names, x_train, y_train)
            self.performance_report_writer.writerow([key, model['train_time']])
            
            # Process each scorer
            for scorer in self.scorers:
                scorer_key = key + '__' + scorer
                candidates = self.refit_candidates(pipeline, model['features'], estimator, scorer, x_train, y_train)
                self.total_fits[estimator] += len(candidates)
                
                # Process each candidate
                for position, candidate in enumerate(candidates):
                    print('\t#%d' % (position+1))
                    
                    # Create base result
                    result = self.create_base_result(
                        scorer_key, estimator, scaler, feature_selector, searcher, scorer, n_classes, position
                    )
                    
                    # Store main model
                    self.main_models[result['key']] = candidate['best_estimator']
                    
                    # Evaluate the model using the base class method
                    evaluation_result = self.evaluate_model_complete(
                        pipeline, model['features'], candidate['best_estimator'],
                        x_val, y_val, x_test, y_test, labels
                    )
                    
                    # Update result with evaluation metrics
                    result.update(evaluation_result)
                    result.update({
                        'selected_features': list(model['selected_features']),
                        'feature_scores': model['feature_scores'],
                        'best_params': candidate['best_params']
                    })
                    
                    # Write result to CSV
                    self.write_result_to_csv(result)
                    
                    # Generate OvR models and results (without retraining - macro mode)
                    print(f"\t\tGenerating OvR metrics for {n_classes} classes...")
                    
                    csv_entries, class_data, new_ovr_models, additional_fits = self._generate_ovr_models_and_results(
                        pipeline, model['features'], candidate['best_estimator'], result,
                        x_train, y_train, x_val, y_val, x_test, y_test, labels,
                        estimator, scorer, reoptimize_ovr=False  # Key difference: always False
                    )
                    
                    # Update total fits with OvR computation fits (should be 0 since no retraining)
                    self.total_fits[estimator] += additional_fits
                    
                    # Store OvR model references (will just point to main model)
                    self.ovr_models.update(new_ovr_models)
                    
                    # Write all OvR CSV entries
                    for csv_entry in csv_entries:
                        self.write_result_to_csv(csv_entry)
                    
                    # Save class-specific data as .pkl.gz file
                    self.save_class_results(class_data, result['key'])
                    
                    print(f"\t\tGenerated {len(csv_entries)} OvR metric entries")
        
        # Finalize reports and save results
        self.finalize_reports(start, n_classes)
        
        print(f"Multiclass macro-averaging classification completed successfully!")
        print(f"Generated {len(self.main_models)} main models and {len(self.ovr_models)} OvR metric entries")
        print(f"Total models: {len(self.main_models)} (OvR entries use same models)")
        
        return True
