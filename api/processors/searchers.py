"""
All hyper-parameter search methods
"""

import os

import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold

from .estimators import ESTIMATORS
from .hyperparameters import HYPER_PARAMETER_RANGE

load_dotenv()
SHUFFLE = False if os.getenv('IGNORE_SHUFFLE', '') != '' else True

# Define the cross validator (shuffle the data between each fold)
# This reduces correlation between outcome and train data order.
CROSS_VALIDATOR = StratifiedKFold(n_splits=10, shuffle=SHUFFLE)

# Define the max iterations for random
MAX_RANDOM_ITERATIONS = 100

def make_grid_search(estimator, scoring, _):
    """Generate grid search with 10 fold cross validator"""

    parameter_range = HYPER_PARAMETER_RANGE['grid'][estimator]\
        if estimator in HYPER_PARAMETER_RANGE['grid'] else {}

    return (
        GridSearchCV(
            ESTIMATORS[estimator],
            parameter_range,
            cv=CROSS_VALIDATOR,
            scoring=scoring,
            n_jobs=-1,
            iid=True,
            return_train_score=False
        ),
        len(list(ParameterGrid(parameter_range))) *\
            CROSS_VALIDATOR.get_n_splits()
    )

def make_random_search(estimator, scoring, y_train):
    """Generate random search with defined max iterations"""

    parameter_range = HYPER_PARAMETER_RANGE['random'][estimator]\
        if estimator in HYPER_PARAMETER_RANGE['random'] else {}

    if callable(parameter_range):
        parameter_range = parameter_range(pd.Series(y_train).value_counts().min())

    total_range = len(list(ParameterGrid(parameter_range)))
    iterations = total_range if MAX_RANDOM_ITERATIONS >= total_range else MAX_RANDOM_ITERATIONS

    return (
        RandomizedSearchCV(
            ESTIMATORS[estimator],
            parameter_range,
            cv=CROSS_VALIDATOR,
            scoring=scoring,
            n_iter=iterations,
            n_jobs=-1,
            iid=True,
            return_train_score=False
        ),
        iterations * CROSS_VALIDATOR.get_n_splits()
    )

SEARCHERS = {
    'grid': make_grid_search,
    'random': make_random_search
}

SEARCHER_NAMES = {
    'grid': 'grid search',
    'random': 'random search'
}
