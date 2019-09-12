"""
Generalization of a provided model using a secondary test set.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score,\
    confusion_matrix, classification_report, f1_score

def generalize(model, pipeline, x2, y2, labels=None):

    """"Generalize method"""

    # If scaling is used in the pipeline, scale the test data
    if 'scaler' in pipeline.named_steps:
        x2 = pipeline.named_steps['scaler'].transform(x2)

    if 'feature_selector' in pipeline.named_steps:
        feature_selector_type = pipeline.named_steps['feature_selector'].__class__.__module__

        if feature_selector_type == 'sklearn.feature_selection.univariate_selection':

            # Identify the selected featured for model provided
            for index, feature in reversed(list(enumerate(model['features'].items()))):

                # Remove the feature if unused from the x2 test data
                if not feature[1]:
                    x2 = np.delete(x2, index, axis=1)

        if feature_selector_type in\
         ('sklearn.decomposition.pca', 'random_forest_importance_select'):
            x2 = pipeline.named_steps['feature_selector'].transform(x2)

    predictions = model['best_estimator'].predict(x2)
    print('\t', classification_report(y2, predictions, target_names=labels).replace('\n', '\n\t'))

    print('\tGeneralization:')

    accuracy = accuracy_score(y2, predictions)
    print('\t\tAccuracy:', accuracy)

    auc = roc_auc_score(y2, predictions)
    print('\t\tAUC:', auc)

    tn, fp, fn, tp = confusion_matrix(y2, predictions).ravel()
    f1 = f1_score(y2, predictions, average='macro')
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    print('\t\tSensitivity:', sensitivity)
    print('\t\tSpecificity:', specificity)
    print('\t\tF1:', f1, '\n')

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
