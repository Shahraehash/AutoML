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

def generalize(features, model, pipeline, x2, y2, labels=None, threshold=.5):
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

    return generalization_report(labels, y2, predictions, probabilities)

def generalize_model(payload, label, folder, threshold=.5):
    data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
    x = data[payload['features']].to_numpy()
    y = data[label]

    pipeline = load(folder + '.joblib')
    probabilities = pipeline.predict_proba(x)[:, 1]
    if threshold == .5:
      predictions = pipeline.predict(x)
    else:
      predictions = (probabilities >= threshold).astype(int)

    return generalization_report(['No ' + label, label], y, predictions, probabilities)

def generalize_ensemble(total_models, job_folder, dataset_folder, label):
    x2, y2, feature_names, _, _ = import_csv(dataset_folder + '/test.csv', label)

    data = pd.DataFrame(x2, columns=feature_names)

    soft_result = predict_ensemble(total_models, data, job_folder, 'soft')
    hard_result = predict_ensemble(total_models, data, job_folder, 'hard')

    # Determine if this is binary or multi-class
    unique_labels = sorted(y2.unique())
    if len(unique_labels) == 2:
        labels = ['No ' + label, label]
    else:
        labels = [f'Class {int(cls)}' for cls in unique_labels]

    return {
        'soft_generalization': generalization_report(labels, y2, soft_result['predicted'], soft_result['probability']),
        'hard_generalization': generalization_report(labels, y2, hard_result['predicted'], hard_result['probability'])
    }

def generalization_report(labels, y2, predictions, probabilities):
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
    
    print(labels)
    print('\t', classification_report(y2, predictions, target_names=labels).replace('\n', '\n\t'))

    print('\tGeneralization:')

    accuracy = accuracy_score(y2, predictions)
    print('\t\tAccuracy:', accuracy)

    n_classes = len(np.unique(y2))
    
    # Binary classification
    if n_classes == 2:
        auc = roc_auc_score(y2, predictions) #WHAT IS THIS?
        print('\t\tBinary AUC:', auc)
        
        roc_auc = roc_auc_score(y2, probabilities)
        print('\t\tROC AUC:', roc_auc)
        
        _, tpr, _ = roc_curve(y2, probabilities)
        
        tn, fp, fn, tp = confusion_matrix(y2, predictions).ravel()
            
    # Multiclass classification    
    else:
        # For multiclass ROC AUC, use the proper probability matrix
        roc_auc = roc_auc_score(y2, probabilities, multi_class='ovr', average='macro')
        print('\t\tROC AUC (macro):', roc_auc)
        
        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(y2, predictions)
        
        fp = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
        fn = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
        tp = (np.diag(cnf_matrix)).astype(float)
        tn = (cnf_matrix.sum() - (fp + fn + tp)).astype(float)       

        tpr = tp / (tp + fn)
      
    mcc = matthews_corrcoef(y2, predictions)
    f1 = f1_score(y2, predictions, average='macro')
    
    # Binary Classsification
    if n_classes == 2:
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        prevalence = (tp + fn) / (len(y2))
        print('\t\tSensitivity:', sensitivity)
        print('\t\tSpecificity:', specificity)
        print('\t\tF1:', f1, '\n')

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

        
    # Multi-class Classification 
    else:
        sensitivity = np.mean(tp / (tp+fn))
        specificity = np.mean(tn / (tn+fp))
        prevalence = np.mean((tp + fn) / (len(y2)))
        print('\t\tSensitivity:', sensitivity)
        print('\t\tSpecificity:', specificity)
        print('\t\tF1:', f1, '\n')
        
        # Sum across classes 
        tp_sum = np.sum(tp)
        tn_sum = np.sum(tn)
        fp_sum = np.sum(fp)
        fn_sum = np.sum(fn)

        return {
            'accuracy': round(accuracy, 4),
            'acc_95_ci': clopper_pearson(tp_sum+tn_sum, len(y2)),
            'mcc': round(mcc, 4),
            'avg_sn_sp': round((sensitivity+specificity)/2, 4),
            'roc_auc': round(roc_auc, 4),
            'roc_auc_95_ci': roc_auc_ci(roc_auc, tpr),
            'f1': round(f1, 4),
            'sensitivity': round(sensitivity, 4),
            'sn_95_ci': clopper_pearson(tp_sum, tp_sum+fn_sum),
            'specificity': round(specificity, 4),
            'sp_95_ci': clopper_pearson(tn_sum, tn_sum+fp_sum),
            'prevalence': round(prevalence, 4),
            'pr_95_ci': clopper_pearson(tp_sum+fn_sum, len(y2)),
            'ppv': round(tp_sum / (tp_sum+fp_sum), 4) if tp_sum+fp_sum > 0 else 0,
            'ppv_95_ci': ppv_95_ci(sensitivity, specificity, tp_sum+fn_sum, fp_sum+tn_sum, prevalence),
            'npv': round(tn_sum / (tn_sum+fn_sum), 4) if tn_sum+fn_sum > 0 else 0,
            'npv_95_ci': npv_95_ci(sensitivity, specificity, tp_sum+fn_sum, fp_sum+tn_sum, prevalence),
            'tn': int(tn_sum),
            'tp': int(tp_sum),
            'fn': int(fn_sum),
            'fp': int(fp_sum)
        }
