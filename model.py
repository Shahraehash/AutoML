# Dependencies
import pandas as pd
import json
from timeit import default_timer as timer

from hyperparameters import hyperParameterRange
from scorers import scorerNames

# Define the generic method to generate the best model for the provided estimator
def generateModel(estimatorName, model, X_train, Y_train, X, Y, X2, Y2, labels=None, scoring='accuracy'):
    start = timer()
    model.fit(X_train, Y_train)

    best_params = performance = selected_features = {}

    if 'selector' in model.named_steps:
        selected_features = pd.Series(model.named_steps['selector'].get_support(), index=list(X))

    if estimatorName in hyperParameterRange:
        performance = pd.DataFrame(model.named_steps['estimator'].cv_results_)[['mean_test_score', 'std_test_score']].sort_values(by='mean_test_score', ascending=False)
        model_best = model.named_steps['estimator'].best_estimator_
        best_params = model.named_steps['estimator'].best_params_

        print('\tBest %s: %.7g (sd=%.7g)'
            % (scorerNames[scoring], performance.iloc[0]['mean_test_score'], performance.iloc[0]['std_test_score']))
        print('\tBest parameters:', json.dumps(best_params, indent=4, sort_keys=True).replace('\n', '\n\t'))
    else:
        print('\tNo hyper-parameters to tune for this estimator\n')
        model_best = model

    train_time = timer() - start
    print('\tTraining time is {:.4f} seconds'.format(train_time), '\n')

    return {
        'best_estimator': model_best,
        'best_params': best_params,
        'best_score': (performance.iloc[0]['mean_test_score'], performance.iloc[0]['std_test_score']) if 'iloc' in performance else None,
        'performance': performance,
        'selected_features': selected_features,
        'train_time': train_time
    }
