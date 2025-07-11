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
import csv
import time
import itertools
import gzip
import pickle
from timeit import default_timer as timer
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy import interpolate
from nyoka import skl_to_pmml, xgboost_to_pmml

from .base_classifier import AutoMLClassifier
from .utils.import_data import import_data
from .utils.preprocess import preprocess
from .utils.utils import model_key_to_name, decimate_points, explode_key
from .utils.stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci
from .utils.summary import print_summary
from .processors.estimators import ESTIMATORS, ESTIMATOR_NAMES, get_xgb_classifier
from .processors.scorers import SCORER_NAMES
from .processors.scalers import SCALERS, SCALER_NAMES
from .processors.feature_selection import FEATURE_SELECTORS, FEATURE_SELECTOR_NAMES
from .processors.searchers import SEARCHER_NAMES, SEARCHERS 
from .processors.debug import Debug


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
    
    def import_csv(self, path, label_column, show_warning=False):
        """Import the specificed sheet"""

        # Read the CSV to memory and drop rows with empty values
        data = pd.read_csv(path)

        # Convert cell values to numerical data and drop invalid data
        data = data.apply(pd.to_numeric, errors='coerce').dropna()

        # Drop the label column from the data
        x = data.drop(label_column, axis=1)

        # Save the label colum values
        y = data[label_column]

        # Grab the feature names
        feature_names = list(x)

        # Convert to NumPy array
        x = x.to_numpy()

        # Get unique labels and label counts
        unique_labels = sorted(y.unique())

        label_counts = {}
        for label in unique_labels:
            label_counts[f'class_{int(label)}_count'] = data[data[label_column] == label].shape[0]
        
        if show_warning:
            for label in unique_labels:
                count = label_counts[f'class_{int(label)}_count']
                print('Class %d Cases: %.7g\n' % (int(label), count))

            # Check for class imbalance in multi-class
            counts = list(label_counts.values())
            min_count, max_count = min(counts), max(counts)
            if min_count / max_count < .5: #NOT SURE WHAT THIS THRESHOLD SHOULD BE?
                print('Warning: Classes are not balanced.')
        
        # Return all class counts as a dictionary for multi-class
        return [x, y, feature_names, label_counts, len(unique_labels)]


    def generalize(self, pipeline, features, model, x_test, y_test, labels=None, threshold=.5, class_index=None):
        """"Generalize method"""

        # Process test data based on pipeline
        x_test = preprocess(features, pipeline, x_test)
        proba = model.predict_proba(x_test)
        
        predictions = model.predict(x_test)
        probabilities = proba

        return self.generalization_report(labels, y_test, predictions, probabilities, class_index)

    def generalization_report(self, labels, y_test, predictions, probabilities, class_index=None):
        
        unique_labels = sorted(y_test.unique())
        labels = [f'Class {int(cls)}' for cls in unique_labels]

        print('\t', classification_report(y_test, predictions, target_names=labels).replace('\n', '\n\t'))

        print('\tGeneralization:')
        # Overall accuracy (same for all classes)
        accuracy = accuracy_score(y_test, predictions)
        print('\t\tOverall Accuracy:', accuracy)
        
        # Calculate confusion matrix for OvR decomposition
        cnf_matrix = confusion_matrix(y_test, predictions)
        
        # Check if we need class-specific calculations for one class only
        if class_index is not None:
            # Single class metrics
            
            # Get class-specific ROC AUC score
            roc_auc_scores = roc_auc_score(y_test, probabilities, multi_class='ovr', average=None)
            roc_auc_class = roc_auc_scores[class_index]
            
            # For OvR: current class vs all others
            tp = cnf_matrix[class_index, class_index]
            fn = np.sum(cnf_matrix[class_index, :]) - tp
            fp = np.sum(cnf_matrix[:, class_index]) - tp
            tn = np.sum(cnf_matrix) - tp - fn - fp
            
            # Calculate metrics for this class
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            prevalence = (tp + fn) / len(y_test)
            
            # Class-specific F1 and MCC
            f1_class = f1_score(y_test, predictions, labels=[class_index], average=None)[0]
            y_binary = (y_test == class_index).astype(int)
            pred_binary = (predictions == class_index).astype(int)
            mcc_class = matthews_corrcoef(y_binary, pred_binary)
            
            # Get class-specific ROC curve for CI calculation
            y_binary_prob = probabilities[:, class_index] if probabilities.ndim > 1 else probabilities
            fpr, tpr_curve, _ = roc_curve(y_binary, y_binary_prob)
            
            return {
                'accuracy': round(accuracy, 4),
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
        
        else:
            # Macro-averaged metrics across all classes
            # Get macro-averaged ROC AUC score
            roc_auc_macro = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
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

    def generate_pipeline(self, scaler, feature_selector, estimator, y_train, scoring=None, searcher='grid', shuffle=True, custom_hyper_parameters=None):
        """Generate the pipeline based on incoming arguments"""

        steps = []

        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        if feature_selector and FEATURE_SELECTORS[feature_selector]:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        steps.append(('debug', Debug()))

        if not scoring:
            scoring = ['accuracy']

        scorers = {}
        for scorer in scoring:
            if scorer == 'roc_auc':
                # Use roc_auc_ovr for multiclass problems
                scorers[scorer] = 'roc_auc_ovr'
            else:
                scorers[scorer] = scorer

        search_step = SEARCHERS[searcher](estimator, scorers, shuffle, custom_hyper_parameters, y_train)

        steps.append(('estimator', search_step[0]))

        return (Pipeline(steps), search_step[1])

    def precision_recall(self, pipeline, features, model, x_test, y_test, class_index=None):
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
            
            if class_index is not None and 0 <= class_index < n_classes:
                # One-vs-Rest for specific class
                actual_class_value = unique_classes[class_index]
                y_binary = (y_test == actual_class_value).astype(int)
                class_scores = scores[:, class_index]
                precision, recall, _ = precision_recall_curve(y_binary, class_scores)
            else:
                # Macro-averaged curve
                precision, recall = self.compute_macro_averaged_curve(
                    y_test, scores, unique_classes, use_proba=False
                )
        
        else:
            # Use predict_proba (Random Forest, etc.)
            probabilities = model.predict_proba(x_test)
            if class_index is not None and 0 <= class_index < n_classes:
                # One-vs-Rest for specific class
                actual_class_value = unique_classes[class_index]
                y_binary = (y_test == actual_class_value).astype(int)
                class_probs = probabilities[:, class_index]
                precision, recall, _ = precision_recall_curve(y_binary, class_probs)
            else:
                # Macro-averaged curve
                precision, recall = self.compute_macro_averaged_curve(
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

    def compute_macro_averaged_curve(self, y_test, scores_or_probs, unique_classes, use_proba=True):
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

    def reliability(self, pipeline, features, model, x_test, y_test, class_index=None):
        """Compute reliability curve and Briar score"""

        # Transform values based on the pipeline
        x_test = preprocess(features, pipeline, x_test)

        n_classes = len(np.unique(y_test))

        if hasattr(model, 'decision_function'):
            probabilities = model.decision_function(x_test)
            exp_scores = np.exp(probabilities - np.max(probabilities, axis=1, keepdims=True))
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
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
            
            # If class_index is specified, return One-vs-Rest curve for that class
            else:
                unique_classes = sorted(np.unique(y_test))
                actual_class_value = unique_classes[class_index]
                y_binary = (y_test == actual_class_value).astype(int)
                class_probs = probabilities[:, class_index]
                fop, mpv = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
                brier_score = brier_score_loss(y_binary, class_probs)
        
        else:
            probabilities = model.predict_proba(x_test)
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
            
            # If class_index is specified, return One-vs-Rest curve for that class
            else:
                unique_classes = sorted(np.unique(y_test))
                actual_class_value = unique_classes[class_index]
                y_binary = (y_test == actual_class_value).astype(int)
                class_probs = probabilities[:, class_index]
                fop, mpv = calibration_curve(y_binary, class_probs, n_bins=10, strategy='uniform')
                brier_score = brier_score_loss(y_binary, class_probs)

        return {
            'brier_score': round(brier_score, 4),
            'fop': [round(num, 4) for num in list(fop)],
            'mpv': [round(num, 4) for num in list(mpv)]
        }

    def roc(self, pipeline, features, model, x_test, y_test, class_index=None):
        """Generate the ROC values"""

        # Transform values based on the pipeline
        x_test = preprocess(features, pipeline, x_test)

        probabilities = model.predict_proba(x_test)
        predictions = model.predict(x_test)
        
        # If class_index is None or invalid, return macro-averaged curve (original behavior)
        if class_index is None:
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

        # If class_index is specified, return One-vs-Rest curve for that class
        else:
            unique_classes = sorted(np.unique(y_test))
            actual_class_value = unique_classes[class_index]
            y_binary = (y_test == actual_class_value).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, probabilities[:, class_index])
            roc_auc = roc_auc_score(y_binary, probabilities[:, class_index])
            
            
        fpr, tpr = decimate_points(
        [round(num, 4) for num in list(fpr)],
        [round(num, 4) for num in list(tpr)]
        )

        return {
            'fpr': list(fpr),
            'tpr': list(tpr),
            'roc_auc': roc_auc
        }

    def compute_class_specific_results(self, pipeline, features, estimator, x_val, y_val, n_classes, model_key, class_idx, x_train=None, y_train=None, x_test_orig=None, y_test_orig=None):
        generalization_data = generalize(pipeline, features, estimator, x_val, y_val, class_index=class_idx)
        reliability_data = reliability(pipeline, features, estimator, x_val, y_val, class_idx)
        precision_data = precision_recall(pipeline, features, estimator, x_val, y_val, class_idx)
        roc_data = roc(pipeline, features, estimator, x_val, y_val, class_idx)
        
        # Compute training ROC AUC for this class if training data is provided
        training_roc_auc = None
        roc_delta = None
        
        if x_train is not None and y_train is not None:
            training_roc_data = roc(pipeline, features, estimator, x_train, y_train, class_idx)
            training_roc_auc = training_roc_data['roc_auc']
            
            # Calculate ROC delta (absolute difference between generalization and training)
            if roc_data['roc_auc'] is not None and training_roc_auc is not None:
                roc_delta = round(abs(roc_data['roc_auc'] - training_roc_auc), 4)
        
        # Compute test ROC metrics if original test data is provided
        test_roc_data = None
        if x_test_orig is not None and y_test_orig is not None:
            test_roc_data = roc(pipeline, features, estimator, x_test_orig, y_test_orig, class_idx)
        
        return {
            'generalization': generalization_data,
            'reliability': reliability_data,
            'precision_recall': precision_data,
            'roc_auc': roc_data,
            'training_roc_auc': training_roc_auc,
            'roc_delta': roc_delta,
            'test_roc_data': test_roc_data
        }

    def generate_ovr_models_and_results(self, pipeline, features, main_model, main_result, x_train, y_train, x_val, y_val, x_test, y_test, labels, estimator, scorer):
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
            y_val_binary = (y_val == actual_class_value).astype(int)
            y_test_binary = (y_test == actual_class_value).astype(int)
            
            # Efficient mode: Use main model
            ovr_model = main_model
            ovr_best_params = main_result['best_params']
        
            # Efficient mode: Use multiclass classification path with class_idx
            class_metrics = self.compute_class_specific_results(
                pipeline, features, ovr_model, x_val, y_val, n_classes, 
                main_result['key'], class_idx, x_train, y_train, x_test, y_test
            )
            
            # Store class data for .pkl.gz file
            all_class_data['class_data'][class_idx] = class_metrics
            
            # Create CSV entry for this OvR model using already computed metrics
            csv_entry = self.create_ovr_csv_entry(
                main_result, class_idx, ovr_best_params, class_metrics
            )
            
            csv_entries.append(csv_entry)
        
        return csv_entries, all_class_data, ovr_models, total_fits

    def create_ovr_csv_entry(self, main_result, class_idx, ovr_best_params, class_metrics):
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
    
    def save_class_results(self, class_data, output_dir, model_key):
        """Save class-specific results with compression"""
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = f"{output_dir}/{model_key}.pkl.gz"
        try:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(class_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving class results for {model_key}: {e}")

    def find_best_model(self, train_set=None, test_set=None, labels=None, label_column=None, parameters=None, output_path='.', update_function=lambda x, y: None):
        """
        Multiclass classification with macro-averaging and class-specific results (no model re-optimization).
        """
        start = timer()     
        
        ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
        ignore_feature_selector = \
            [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
        ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
        ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
        shuffle = False if parameters.get('ignore_shuffle', '') != '' else True
        scorers = [x for x in SCORER_NAMES if x not in \
            [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]]

        # Basic validation
        if train_set is None:
            print('Missing training data')
            return {}

        if test_set is None:
            print('Missing test data')
            return {}

        if label_column is None:
            print('Missing column name for classifier target')
            return {}

        custom_hyper_parameters = json.loads(parameters['hyper_parameters'])\
            if 'hyper_parameters' in parameters else None

        # Import data
        (x_train, x_val, y_train, y_val, x_test, y_test, feature_names, metadata) = \
            import_data(train_set, test_set, label_column)

        # Determine number of classes - must be multiclass
        n_classes = len(np.unique(y_train))
        print(f'Detected {n_classes} classes in target column')
        
        if n_classes <= 2:
            raise ValueError(f"MulticlassMacroClassifier expects more than 2 classes, got {n_classes}")

        total_fits = {}
        csv_header_written = False

        # Memory-efficient model storage
        main_models = {}  # Store main models
        ovr_models = {}   # Store references to main models (no separate training)

        all_pipelines = list(itertools.product(*[
            filter(lambda x: False if x in ignore_estimator else True, ESTIMATOR_NAMES),
            filter(lambda x: False if x in ignore_scaler else True, SCALER_NAMES),
            filter(lambda x: False if x in ignore_feature_selector else True, FEATURE_SELECTOR_NAMES),
            filter(lambda x: False if x in ignore_searcher else True, SEARCHER_NAMES),
        ]))

        if not len(all_pipelines):
            print('No pipelines to run with the current configuration')
            return False

        report = open(output_path + '/report.csv', 'w+')
        report_writer = csv.writer(report)

        performance_report = open(output_path + '/performance_report.csv', 'w+')
        performance_report_writer = csv.writer(performance_report)
        performance_report_writer.writerow(['key', 'train_time (s)'])

        print(f"Starting multiclass macro-averaging classification with {len(all_pipelines)} pipeline combinations...")
        print(f"Number of classes: {n_classes}")

        for index, (estimator, scaler, feature_selector, searcher) in enumerate(all_pipelines):

            # Trigger callback for task monitoring
            update_function(index, len(all_pipelines))

            key = '__'.join([scaler, feature_selector, estimator, searcher])
            print('Generating ' + model_key_to_name(key))

            # Generate the pipeline
            pipeline = self.generate_pipeline(scaler, feature_selector, estimator, y_train, scorers, searcher, shuffle, custom_hyper_parameters)

            if not estimator in total_fits:
                total_fits[estimator] = 0
            total_fits[estimator] += pipeline[1]

            # Fit the pipeline
            model = self.generate_model(pipeline[0], feature_names, x_train, y_train)
            performance_report_writer.writerow([key, model['train_time']])

            for scorer in scorers:
                scorer_key = key + '__' + scorer
                candidates = self.refit_model(pipeline[0], model['features'], estimator, scorer, x_train, y_train)
                total_fits[estimator] += len(candidates)

                for position, candidate in enumerate(candidates):
                    print('\t#%d' % (position+1))
                    
                    # Create base result
                    result = {
                        'key': scorer_key + '__' + str(position),
                        'class_type': 'multiclass',
                        'class_index': None,  # Main model has no specific class
                        'scaler': SCALER_NAMES[scaler],
                        'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
                        'algorithm': ESTIMATOR_NAMES[estimator],
                        'searcher': SEARCHER_NAMES[searcher],
                        'scorer': SCORER_NAMES[scorer],
                    }

                    # Store main model
                    main_models[result['key']] = candidate['best_estimator']

                    # Update result with evaluation metrics
                    result.update(self.generalize(pipeline[0], model['features'], candidate['best_estimator'], x_val, y_val, labels))
                    result.update({
                        'selected_features': list(model['selected_features']),
                        'feature_scores': model['feature_scores'],
                        'best_params': candidate['best_params']
                    })

                    roc_auc = self.roc(pipeline[0], model['features'], candidate['best_estimator'], x_val, y_val)
                    result.update({
                        'test_fpr': roc_auc['fpr'],
                        'test_tpr': roc_auc['tpr'],
                        'training_roc_auc': roc_auc['roc_auc']
                    })
                    result['roc_delta'] = round(abs(result['roc_auc'] - result['training_roc_auc']), 4)
                    roc_auc = self.roc(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test)
                    result.update({
                        'generalization_fpr': roc_auc['fpr'],
                        'generalization_tpr': roc_auc['tpr']
                    })
                    result.update(self.reliability(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test))
                    result.update(self.precision_recall(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test))
                    
                    # Write main model result
                    if not csv_header_written:
                        report_writer.writerow(result.keys())
                        csv_header_written = True

                    report_writer.writerow(list([str(i) for i in result.values()]))

                    # Generate OvR metrics for each class (no retraining - macro mode)
                    print(f"\t\tGenerating OvR metrics for {n_classes} classes...")
                    
                    for class_idx, class_label in enumerate(labels):
                        # Use the simplified function to handle all OvR logic
                        csv_entries, class_data, new_ovr_models, additional_fits = self.generate_ovr_models_and_results(
                            pipeline[0], model['features'], candidate['best_estimator'], result,
                            x_train, y_train, x_val, y_val, x_test, y_test, labels,
                            estimator, scorer
                        )
                        
                        # Update total fits
                        total_fits[estimator] += additional_fits
                        
                        # Store any new OvR models
                        ovr_models.update(new_ovr_models)
                        
                        # Write all CSV entries
                        for csv_entry in csv_entries:
                            report_writer.writerow(list([str(i) for i in csv_entry.values()]))
                        
                        # Save class-specific data as .pkl.gz file
                        class_results_dir = output_path + '/class_results'
                        self.save_class_results(class_data, class_results_dir, result['key'])

                        print(f"\t\tGenerated {n_classes} OvR metric entries")

        train_time = timer() - start
        print('\tTotal run time is {:.4f} seconds'.format(train_time), '\n')
        performance_report_writer.writerow(['total', train_time])

        report.close()
        performance_report.close()
        print('Total fits generated', sum(total_fits.values()))
        print_summary(output_path + '/report.csv')
        
        self.save_model_archives(main_models, ovr_models, output_path)

        # Update the metadata and write it out
        metadata.update({
            'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'fits': total_fits,
            'ovr_reoptimized': False,
            'ovr_models_count': len(ovr_models),
            'main_models_count': len(main_models),
            'n_classes': n_classes
        })

        if output_path != '.':
            metadata_path = output_path + '/metadata.json'
            # Check if metadata file exists and load existing data
            existing_metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as metafile:
                    existing_metadata = json.load(metafile)
            
                # Update with new metadata
                existing_metadata.update(metadata)
            
                # Write the updated metadata
                with open(metadata_path, 'w') as metafile:
                    json.dump(existing_metadata, metafile, indent=2)
        
        return True

    @staticmethod
    def create_model(key, hyper_parameters, selected_features, dataset_path=None, label_column=None, output_path='.', threshold=.5):
        """Refits the requested model and pickles it for export"""

        if dataset_path is None:
            print('Missing dataset path')
            return {}

        if label_column is None:
            print('Missing column name for classifier target')
            return {}

        # Import data
        (x_train, _, y_train, _, x_test, y_test, features, _) = \
            import_data(dataset_path + '/train.csv', dataset_path + '/test.csv', label_column)

        # Get pipeline details from the key
        scaler, feature_selector, estimator, _, _ = explode_key(key)
        steps = []

        # Drop the unused features
        if 'pca-' not in feature_selector:
            for index, feature in reversed(list(enumerate(features))):
                if feature not in selected_features:
                    x_train = np.delete(x_train, index, axis=1)
                    x_test = np.delete(x_test, index, axis=1)

        # Add the scaler, if used
        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        # Add the feature transformer
        if 'pca-' in feature_selector:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        # Add the estimator with proper XGBoost configuration
        if estimator == 'gb':
            n_classes = len(pd.Series(y_train).unique())
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

        unique_labels = sorted(y_test.unique())
        labels = [f'Class {int(cls)}' for cls in unique_labels]

        # Assess the model performance and store the results
        generalization_result = self.evaluate_generalization(pipeline, model['features'], pipeline['estimator'], x_test, y_test, labels)
        with open(output_path + '/pipeline.json', 'w') as statsfile:
            json.dump(generalization_result, statsfile)

        # Dump the pipeline to a file
        dump(pipeline, output_path + '/pipeline.joblib')
        pd.DataFrame([selected_features]).to_csv(output_path + '/input.csv', index=False, header=False)

        # Export the model as a PMML
        try:
            if estimator == 'gb':
                xgboost_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
            else:
                skl_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
        except Exception:
            try:
                os.remove(output_path + '/pipeline.pmml')
            except OSError:
                pass

        return generalization_result

    @staticmethod
    def generalize_model(payload, label, folder, threshold=.5):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]
        unique_labels = sorted(y.unique())
        labels = [f'Class {int(cls)}' for cls in unique_labels]
        
        pipeline = load(folder + '.joblib')
        probabilities = pipeline.predict_proba(x)[:, 1]
        if threshold == .5:
            predictions = pipeline.predict(x)
        else:
            predictions = (probabilities >= threshold).astype(int)

        return generalization_report(labels, y, predictions, probabilities)

    @staticmethod
    def generalize_ensemble(total_models, job_folder, dataset_folder, label):
        x_test, y_test, feature_names, _, _ = import_csv(dataset_folder + '/test.csv', label)

        data = pd.DataFrame(x_test, columns=feature_names)

        soft_result = self.predict_ensemble(total_models, data, job_folder, 'soft')
        hard_result = self.predict_ensemble(total_models, data, job_folder, 'hard')

        unique_labels = sorted(y_test.unique())
        labels = [f'Class {int(cls)}' for cls in unique_labels]

        return {
            'soft_generalization': generalization_report(labels, y_test, soft_result['predicted'], soft_result['probability']),
            'hard_generalization': generalization_report(labels, y_test, hard_result['predicted'], hard_result['probability'])
        }

    @staticmethod
    def additional_precision(payload, label, folder, class_index=None):
        """Return additional precision recall curve"""

        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.precision_recall(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)
    
    @staticmethod
    def additional_reliability(payload, label, folder, class_index=None):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.reliability(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)

    @staticmethod
    def additional_roc(payload, label, folder, class_index=None):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.roc(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)


    @staticmethod
    def predict(data, path='.', threshold=.5):
        """Predicts against the provided data"""

        # Load the pipeline
        pipeline = load(path + '.joblib')

        data = pd.DataFrame(data).dropna().values

        probability = pipeline.predict_proba(data)
        if threshold == .5:
            predicted = pipeline.predict(data)
        else:
            predicted = (probability[:, 1] >= threshold).astype(int)

        return {
            'predicted': predicted.tolist(),
            'probability': [sublist[predicted[index]] for index, sublist in enumerate(probability.tolist())]
        }

    @staticmethod
    def predict_ensemble(total_models, data, path='.', vote_type='soft'):
        """Predicts against the provided data by creating an ensemble of the selected models"""

        probabilities = []
        predictions = []

        for x in range(total_models):
            pipeline = load(path + '/ensemble' + str(x) + '.joblib')

            with open(path + '/ensemble' + str(x) + '_features.json') as feature_file:
                features = json.load(feature_file)

            selected_data = data[features].dropna().to_numpy()
            probabilities.append(pipeline.predict_proba(selected_data))
            predictions.append(pipeline.predict(selected_data))

        predictions = np.asarray(predictions).T
        probabilities = np.average(np.asarray(probabilities), axis=0)

        if vote_type == 'soft':
            predicted = np.argmax(probabilities, axis=1)
        else:
            predicted = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions.astype('int')
            )

        return {
            'predicted': predicted.tolist(),
            'probability': [sublist[predicted[index]] for index, sublist in enumerate(probabilities.tolist())]
        }
