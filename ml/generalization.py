"""
Generalization of a provided model using a secondary test set.
"""

from joblib import load
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score,\
    confusion_matrix, classification_report, f1_score, roc_curve

from .preprocess import preprocess
from .predict import predict_ensemble
from .import_data import import_csv
from .stats import clopper_pearson, roc_auc_ci

def generalize(features, model, pipeline, x2, y2, labels=None, threshold=.5):
    """"Generalize method"""

    # Process test data based on pipeline
    x2 = preprocess(features, pipeline, x2)
    probabilities = model.predict_proba(x2)[:, 1]
    if threshold == .5:
      predictions = model.predict(x2)
    else:
      predictions = (probabilities >= threshold).astype(int)

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

    return {
        'soft_generalization': generalization_report(['No ' + label, label], y2, soft_result['predicted'], soft_result['probability']),
        'hard_generalization': generalization_report(['No ' + label, label], y2, hard_result['predicted'], hard_result['probability'])
    }

def generalization_report(labels, y2, predictions, probabilities):
    print('\t', classification_report(y2, predictions, target_names=labels).replace('\n', '\n\t'))

    print('\tGeneralization:')

    accuracy = accuracy_score(y2, predictions)
    print('\t\tAccuracy:', accuracy)

    auc = roc_auc_score(y2, predictions)
    print('\t\tBinary AUC:', auc)

    roc_auc = roc_auc_score(y2, probabilities)
    print('\t\tROC AUC:', roc_auc)

    _, tpr, _ = roc_curve(y2, probabilities)

    tn, fp, fn, tp = confusion_matrix(y2, predictions).ravel()
    f1 = f1_score(y2, predictions, average='macro')
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    prevalence = (tp + fn) / (len(y2))
    print('\t\tSensitivity:', sensitivity)
    print('\t\tSpecificity:', specificity)
    print('\t\tF1:', f1, '\n')

    return {
        'accuracy': round(accuracy, 4),
        'acc_95_ci': clopper_pearson(tp+tn, len(y2)),
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
        'npv': round(tn / (tn+fn), 4) if tn+fn > 0 else 0,
        'tn': int(tn),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp)
    }
