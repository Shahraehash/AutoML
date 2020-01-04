"""
Processes configuration to generate all pipelines
"""

import itertools

from .processors.estimators import ESTIMATOR_NAMES
from .processors.feature_selection import FEATURE_SELECTOR_NAMES
from .processors.scalers import SCALER_NAMES
from .processors.searchers import SEARCHER_NAMES
from .processors.scorers import SCORER_NAMES

def list_pipelines(parameters):
    """Construct array of all pipeline combinations"""

    ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
    ignore_feature_selector = \
        [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
    ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
    ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
    scorers = [x for x in SCORER_NAMES if x not in \
        [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]]

    return list(filter(filter_invalid_svm_pipelines, itertools.product(*[
        filter(lambda x: False if x in ignore_estimator else True, ESTIMATOR_NAMES),
        filter(lambda x: False if x in ignore_scaler else True, SCALER_NAMES),
        filter(lambda x: False if x in ignore_feature_selector else True, FEATURE_SELECTOR_NAMES),
        filter(lambda x: False if x in ignore_searcher else True, SEARCHER_NAMES),
        scorers
    ])))

def filter_invalid_svm_pipelines(pipeline):
    """
    SVM without robust scaling can loop consuming infinite CPU
    time so we prevent any other combination here.
    """

    if pipeline[0] == 'svm' and\
        (
            pipeline[1] == 'none' or\
            pipeline[1] == 'std' and pipeline[3] == 'random'
        ):
        return False
    return True
