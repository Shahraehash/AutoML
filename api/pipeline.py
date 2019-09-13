"""
Generates a pipeline
"""

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid

from .processors.debug import Debug
from .processors.estimators import ESTIMATORS
from .processors.feature_selection import FEATURE_SELECTORS
from .processors.scalers import SCALERS
from .hyperparameters import HYPER_PARAMETER_RANGE

# Define the cross validator
CROSS_VALIDATOR = StratifiedKFold(n_splits=10)

# Generate a pipeline
def generate_pipeline(scaler, feature_selector, estimator, scoring='accuracy'):
    """Generate the pipeline based on incoming arguments"""

    steps = []

    if scaler and SCALERS[scaler]:
        steps.append(('scaler', SCALERS[scaler]))

    if feature_selector and FEATURE_SELECTORS[feature_selector]:
        steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

    steps.append(('debug', Debug()))

    if estimator in HYPER_PARAMETER_RANGE:
        steps.append(('estimator', GridSearchCV(
            ESTIMATORS[estimator],
            HYPER_PARAMETER_RANGE[estimator],
            return_train_score='False',
            cv=CROSS_VALIDATOR,
            n_jobs=-1,
            iid=True,
            scoring=scoring
        )))

        total_fits = len(ParameterGrid(HYPER_PARAMETER_RANGE[estimator])) *\
            CROSS_VALIDATOR.get_n_splits()
    else:
        steps.append(('estimator', ESTIMATORS[estimator]))
        total_fits = CROSS_VALIDATOR.get_n_splits()

    return (Pipeline(steps), total_fits)
