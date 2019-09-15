"""
All hyper-parameter search methods
"""

from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold

from .estimators import ESTIMATORS
from .hyperparameters import HYPER_PARAMETER_RANGE

# Define the cross validator
CROSS_VALIDATOR = StratifiedKFold(n_splits=10, shuffle=True)

# Define the max iterations for random
MAX_RANDOM_ITERATIONS = 100

def make_grid_search(estimator, scoring):
    """Generate grid search with 10 fold cross validator"""

    if estimator not in HYPER_PARAMETER_RANGE['grid']:
        return (ESTIMATORS[estimator], 1)

    return (
        GridSearchCV(
            ESTIMATORS[estimator],
            HYPER_PARAMETER_RANGE['grid'][estimator],
            cv=CROSS_VALIDATOR,
            scoring=scoring,
            n_jobs=-1,
            iid=True,
            return_train_score=False
        ),
        len(ParameterGrid(HYPER_PARAMETER_RANGE['grid'][estimator])) *\
            CROSS_VALIDATOR.get_n_splits()
    )

def make_random_search(estimator, scoring):
    """Generate random search with defined max iterations"""

    if estimator not in HYPER_PARAMETER_RANGE['random']:
        return (ESTIMATORS[estimator], 1)

    return (
        RandomizedSearchCV(
            ESTIMATORS[estimator],
            HYPER_PARAMETER_RANGE['random'][estimator],
            cv=CROSS_VALIDATOR,
            scoring=scoring,
            n_iter=MAX_RANDOM_ITERATIONS,
            n_jobs=-1,
            iid=True,
            return_train_score=False
        ),
        MAX_RANDOM_ITERATIONS * CROSS_VALIDATOR.get_n_splits()
    )

SEARCHERS = {
    'grid': make_grid_search,
    'random': make_random_search
}

SEARCHER_NAMES = {
    'grid': 'grid search',
    'random': 'random search'
}
