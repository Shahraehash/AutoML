"""
Generalization of a provided model using a secondary test set.
"""

from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score,\
    confusion_matrix, classification_report, f1_score, roc_curve,\
    matthews_corrcoef

from .preprocess import preprocess
from .predict import predict_ensemble
from .import_data import import_csv
from .stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci

def generalize(pipeline, features, model, x2, y2, labels=None, threshold=.5, class_index=None):
    """"Generalize method"""

    # Process test data based on pipeline
    x2 = preprocess(features, pipeline, x2)
    proba = model.predict_proba(x2)
    
    # Binary classification
    if proba.shape[1] == 2:
        probabilities = proba[:, 1]
        if threshold == .5:
            predictions = model.predict(x2)
        else:
            predictions = (probabilities >= threshold).astype(int)
        
    # Multiclass classification
    else:
        predictions = model.predict(x2)
        probabilities = proba

    return generalization_report(labels, y2, predictions, probabilities, class_index)

def generalize_model(payload, label, folder, threshold=.5):
    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    x = data[payload['features']].to_numpy()
    y = data[label]

    pipeline = load(folder + '.joblib')
    proba = pipeline.predict_proba(x)
    # Binary classification
    if proba.shape[1] == 2:
        probabilities = proba[:, 1]
        if threshold == .5:
            predictions = pipeline.predict(x)
        else:
            predictions = (probabilities >= threshold).astype(int)
        labels = ['No ' + label, label]
    
    # Multiclass classification
    else:
        predictions = pipeline.predict(x)
        probabilities = proba
        # Generate appropriate labels for multi-class
        unique_classes = sorted(np.unique(y))
        labels = [f'Class {int(cls)}' for cls in unique_classes]

    return generalization_report(['No ' + label, label], y, predictions, probabilities)

def generalize_ensemble(total_models, job_folder, dataset_folder, label):
    x2, y2, feature_names, _, _, label_mapping_info = import_csv(dataset_folder + '/test.csv', label)

    data = pd.DataFrame(x2, columns=feature_names)

    soft_result = predict_ensemble(total_models, data, job_folder, 'soft')
    hard_result = predict_ensemble(total_models, data, job_folder, 'hard')

    # Determine if this is binary or multi-class and generate appropriate labels
    if label_mapping_info and 'original_labels' in label_mapping_info:
        original_labels = label_mapping_info['original_labels']
        if len(original_labels) == 2:
            labels = ['No ' + label, label]
        else:
            labels = [f'Class {int(cls)}' for cls in original_labels]
    else:
        # Fallback to current logic for backward compatibility
        unique_labels = sorted(y2.unique())
        if len(unique_labels) == 2:
            labels = ['No ' + label, label]
        else:
            labels = [f'Class {int(cls)}' for cls in unique_labels]

    return {
        'soft_generalization': generalization_report(labels, y2, soft_result['predicted'], soft_result['probability']),
        'hard_generalization': generalization_report(labels, y2, hard_result['predicted'], hard_result['probability'])
    }

def generalization_report(labels, y2, predictions, probabilities, class_index=None):
    # Get unique classes from actual data
    unique_classes = sorted(np.unique(y2))
    n_classes = len(unique_classes)
    
    # If labels are provided but don't match the number of classes, generate appropriate labels
    if labels is None or len(labels) != n_classes:
        if n_classes == 2:
            # For binary classification, use generic labels if not provided correctly
            labels = ['Class 0', 'Class 1']
        else:
            # For multiclass, generate labels based on actual class values
            labels = [f'Class {int(cls)}' for cls in unique_classes]
    
    print('\t', classification_report(y2, predictions, target_names=labels).replace('\n', '\n\t'))

    print('\tGeneralization:')
    n_classes = len(np.unique(y2))
    
    # Binary classification
    if n_classes == 2:
        accuracy = accuracy_score(y2, predictions)
        print('\t\tAccuracy:', accuracy)

        auc = roc_auc_score(y2, predictions)
        
        roc_auc = roc_auc_score(y2, probabilities)
        print('\t\tROC AUC:', roc_auc)
        
        _, tpr, _ = roc_curve(y2, probabilities)
        
        tn, fp, fn, tp = confusion_matrix(y2, predictions).ravel()
            
        mcc = matthews_corrcoef(y2, predictions)
        f1 = f1_score(y2, predictions)

        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        prevalence = (tp + fn) / (len(y2))

        return {
            'accuracy': round(accuracy, 4),
            'acc_95_ci': clopper_pearson(tp+tn, len(y2)),
            'mcc': round(mcc, 4),
            'avg_sn_sp': round(auc, 4),
            'roc_auc': round(roc_auc, 4),
            'roc_auc_95_ci': roc_auc_ci(roc_auc, tpr),
            'f1': round(f1, 4),
            'sensitivity': round(sensitivity, 4),
            'sn_95_ci': clopper_pearson(tp, tp+fn),
            'specificity': round(specificity, 4),
            'sp_95_ci': clopper_pearson(tn, tn+fp),
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

    # Multi-class Classification - Class-specific metrics using OvR
    else:
        # Overall accuracy (same for all classes)
        accuracy = accuracy_score(y2, predictions)
        print('\t\tOverall Accuracy:', accuracy)
        
        # Calculate confusion matrix for OvR decomposition
        cnf_matrix = confusion_matrix(y2, predictions)
        
        # Check if we need class-specific calculations for one class only
        if class_index is not None:
            # Single class metrics
            
            # Get class-specific ROC AUC score
            roc_auc_scores = roc_auc_score(y2, probabilities, multi_class='ovr', average=None)
            roc_auc_class = roc_auc_scores[class_index]
            
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
                'accuracy': round(accuracy, 4),
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
