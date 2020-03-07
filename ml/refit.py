"""
Refit a model based on the provided parameters
and score it against the test set for the provided
scoring method.
"""

import json
from sklearn.base import clone

from .processors.estimators import ESTIMATORS
from .processors.scorers import SCORER_NAMES
from .preprocess import preprocess

MODELS_TO_EVALUATE = 2

def refit_model(pipeline, features, estimator, scoring, x_train, y_train):
    """
    Determine the best model based on the provided scoring method
    and the fitting pipeline results.
    """

    # Transform values based on the pipeline
    x_train = preprocess(features, pipeline, x_train)

    results = pipeline.named_steps['estimator'].cv_results_

    # Select the top search results
    sorted_results = sorted(
        range(len(results['rank_test_%s' % scoring])),
        key=lambda i: results['rank_test_%s' % scoring][i]
    )[:MODELS_TO_EVALUATE]

    models = []

    for position, index in enumerate(sorted_results):
        best_params_ = results['params'][index]

        print('\t#%d %s: %.7g (sd=%.7g)'
              % (position+1, SCORER_NAMES[scoring], results['mean_test_%s' % scoring][index],
                 results['std_test_%s' % scoring][index]))
        print('\t#%d %s parameters:' % (position+1, SCORER_NAMES[scoring]),
              json.dumps(best_params_, indent=4, sort_keys=True).replace('\n', '\n\t'))

        model = clone(ESTIMATORS[estimator]).set_params(
            **best_params_).fit(x_train, y_train)

        models.append({
            'best_estimator': model,
            'best_params': best_params_
        })

    return models
