"""
Generates a pipeline
"""

from sklearn.pipeline import Pipeline

from .processors.debug import Debug
from .processors.feature_selection import FEATURE_SELECTORS
from .processors.scalers import SCALERS
from .processors.searchers import SEARCHERS

# Generate a pipeline
def generate_pipeline(scaler, feature_selector, estimator, y_train, scoring='accuracy', searcher='grid'):
    """Generate the pipeline based on incoming arguments"""

    steps = []

    if scaler and SCALERS[scaler]:
        steps.append(('scaler', SCALERS[scaler]))

    if feature_selector and FEATURE_SELECTORS[feature_selector]:
        steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

    steps.append(('debug', Debug()))

    search_step = SEARCHERS[searcher](estimator, scoring, y_train)

    steps.append(('estimator', search_step[0]))

    return (Pipeline(steps), search_step[1])
