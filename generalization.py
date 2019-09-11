# Dependencies
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, f1_score

from scalers import scalers

def generalize(model, pipeline, X2, Y2, labels=None):

    # If scaling is used in the pipeline, scale the test data
    if 'scaler' in pipeline.named_steps:
        X2 = pipeline.named_steps['scaler'].transform(X2)
    else:
        X2 = np.array(X2)

    if 'feature_selector' in pipeline.named_steps:
        featureSelectorType = pipeline.named_steps['feature_selector'].__class__.__module__

        if featureSelectorType == 'sklearn.feature_selection.univariate_selection':

            # Identify the selected featured for model provided
            for index, feature in reversed(list(enumerate(model['features'].items()))):

                # Remove the feature if unused from the X2 test data
                if not feature[1]:
                    X2 = np.delete(X2, index, axis=1)

        if featureSelectorType == 'sklearn.decomposition.pca':
            X2 = pipeline.named_steps['feature_selector'].transform(X2)

    predictions = model['best_estimator'].predict(X2)
    print('\t', classification_report(Y2, predictions, target_names=labels).replace('\n', '\n\t'))

    print('\tGeneralization:')

    accuracy = accuracy_score(Y2, predictions)
    print('\t\tAccuracy:', accuracy)

    auc = roc_auc_score(Y2, predictions)
    print('\t\tAUC:', auc)

    tn, fp, fn, tp = confusion_matrix(Y2, predictions).ravel()
    f1 = f1_score(Y2, predictions, average='macro')
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