"""
All hyper-parameter search methods
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid

from .estimators import ESTIMATORS
from ..hyperparameters import HYPER_PARAMETER_RANGE

# Define the cross validator
CROSS_VALIDATOR = StratifiedKFold(n_splits=10)

def make_grid_search(estimator, scoring):
    if estimator not in HYPER_PARAMETER_RANGE:
        return (ESTIMATORS[estimator], 1)

    total_fits = len(ParameterGrid(HYPER_PARAMETER_RANGE[estimator])) *\
        CROSS_VALIDATOR.get_n_splits()

    return (GridSearchCV(
        ESTIMATORS[estimator],
        HYPER_PARAMETER_RANGE[estimator],
        return_train_score='False',
        cv=CROSS_VALIDATOR,
        n_jobs=-1,
        iid=True,
        scoring=scoring
    ), total_fits)

SEARCHERS = {
    'grid': make_grid_search
}

SEARCHER_NAMES = {
    'grid': 'grid search'
}
