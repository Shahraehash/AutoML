# Dependencies
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, f1_score

from scalers import scalers

def generalize(model, scaler, X_train, X2, Y2, labels=None):

    # Identify the selected featured for model provided
    for feature, selected in model['selected_features'].items():

        # Remove the selected fields from X_train (used to generate the scaler)
        # and also remove them from the X2 test data.
        if not selected:
            X_train = X_train.drop(feature, axis=1)
            X2 = X2.drop(feature, axis=1)

    # If scaling is used in the pipeline, scale the test data
    if scalers[scaler]:
        sc = scalers[scaler]
        sc.fit(X_train)
        X2 = sc.transform(X2)

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