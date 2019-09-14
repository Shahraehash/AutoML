"""
Utilities
"""

from .hyperparameters import HYPER_PARAMETER_RANGE
from .processors.estimators import ESTIMATOR_NAMES
from .processors.feature_selection import FEATURE_SELECTOR_NAMES
from .processors.scalers import SCALER_NAMES
from .processors.searchers import SEARCHER_NAMES
from .processors.scorers import SCORER_NAMES

def model_key_to_name(key):
    """"Resolve key name to descriptive name"""

    scaler, feature_selector, estimator, scorer, searcher = key.split('__')

    search_method = ' using '

    if searcher == 'grid':
        if estimator in HYPER_PARAMETER_RANGE:
            search_method += SCORER_NAMES[scorer] + ' scored ' + SEARCHER_NAMES[searcher]
        else:
            search_method = ''
    else:
        search_method += SEARCHER_NAMES[searcher]

    return ESTIMATOR_NAMES[estimator] + ' model' + search_method + ' with ' +\
        SCALER_NAMES[scaler] + ' and with ' + FEATURE_SELECTOR_NAMES[feature_selector]
