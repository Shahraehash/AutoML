# Dependencies
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from estimators import estimators, estimatorNames
from hyperparameters import hyperParameterRange

# Define the cross validator
cv = StratifiedKFold(n_splits=10)

# Define the generic method to generate the best model for the provided estimator
def generateModel(algorithm, X_train, Y_train, X, Y, X2, Y2, labels):
    print('\tGenerating ' + estimatorNames[algorithm] + ' model:')

    model = estimators[algorithm]()
    model.fit(X_train, Y_train)
    model_cv = cross_val_score(model, X, Y, cv=cv, scoring='accuracy')
    best_params = {}
    performance = {}

    print("\t\tDefault CV Accuracy: %.7g (sd=%.7g)" % (np.mean(model_cv), np.std(model_cv)))

    # Perform a grid search if the algorithm has tunable hyper-parameters
    if algorithm in hyperParameterRange:

        # The parameter `return_train_score` is False because 
        # it's not required and reduces CPU time without it
        model_gs = GridSearchCV(
            model,
            hyperParameterRange[algorithm],
            return_train_score='False',
            cv=cv,
            n_jobs=-1,
            scoring='accuracy'
        )
        model_gs.fit(X_train, Y_train)
        model_gs_cv = cross_val_score(model_gs.best_estimator_, X, Y, cv=cv, scoring='roc_auc')

        performance = pd.DataFrame(model_gs.cv_results_)[['mean_test_score', 'std_test_score']].sort_values(by='mean_test_score', ascending=False)
        model_best = model_gs.best_estimator_
        best_params = model_gs.best_params_

        print("\t\tGridSearchCV AUC: %.7g (sd=%.7g)" % (np.mean(model_gs_cv), np.std(model_gs_cv)))
        print('\t\tBest accuracy: %.7g (sd=%.7g)'
            % (performance.iloc[0]['mean_test_score'], performance.iloc[0]['std_test_score']))
        print('\t\tBest parameters:', json.dumps(best_params, indent=4, sort_keys=True).replace('\n', '\n\t\t'))
    else:
        print('\t\tNo hyper-parameters to tune for this estimator\n')
        model_best = model

    predictions = model_best.predict(X2)
    print('\t\t', classification_report(Y2, predictions, target_names=labels).replace('\n', '\n\t\t'))

    accuracy = accuracy_score(Y2, predictions)
    print('\t\tGeneralization accuracy:', accuracy)

    auc = roc_auc_score(Y2, predictions)
    print('\t\tGeneralization AUC:', auc, '\n')

    return {
        'estimator': model_best,
        'best_params': best_params,
        'performance': performance,
        'generalization': {
            'accuracy': accuracy,
            'auc': auc
        }
    }
