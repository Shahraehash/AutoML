"""
Processes configuration to generate all pipelines
"""

import itertools

from ..processors.estimators import ESTIMATOR_NAMES
from ..processors.feature_selection import FEATURE_SELECTOR_NAMES
from ..processors.scalers import SCALER_NAMES
from ..processors.searchers import SEARCHER_NAMES
from ..processors.scorers import SCORER_NAMES

def list_pipelines(parameters):
    """Construct array of all pipeline combinations"""

    ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
    ignore_feature_selector = \
        [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
    ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
    ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
    ignore_scorer = [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]

    return list(itertools.product(*[
        dict(filter(
            lambda x: False if x[0] in ignore_estimator else True,
            ESTIMATOR_NAMES.items()
        )).values(),
        dict(filter(
            lambda x: False if x[0] in ignore_scaler else True,
            SCALER_NAMES.items()
        )).values(),
        dict(filter(
            lambda x: False if x[0] in ignore_feature_selector else True,
            FEATURE_SELECTOR_NAMES.items()
        )).values(),
        dict(filter(
            lambda x: False if x[0] in ignore_searcher else True,
            SEARCHER_NAMES.items()
        )).values(),
        dict(filter(
            lambda x: False if x[0] in ignore_scorer else True,
            SCORER_NAMES.items()
        )).values(),
    ]))
