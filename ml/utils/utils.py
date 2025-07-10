"""
Utilities
"""

import numpy as np

from ..processors.estimators import ESTIMATOR_NAMES
from ..processors.feature_selection import FEATURE_SELECTOR_NAMES
from ..processors.scalers import SCALER_NAMES
from ..processors.searchers import SEARCHER_NAMES
from ..processors.scorers import SCORER_NAMES

def model_key_to_name(key):
    """"Resolve key name to descriptive name"""

    scaler, feature_selector, estimator, searcher, scorer = explode_key(key)
    search = 'cross validation' if estimator == 'nb' else SEARCHER_NAMES[searcher]
    if scorer:
        search = SCORER_NAMES[scorer] + ' scored ' + search

    return ESTIMATOR_NAMES[estimator] + ' model using ' + search + ' with ' +\
        SCALER_NAMES[scaler] + ' and with ' + FEATURE_SELECTOR_NAMES[feature_selector]

def explode_key(key):
    """Split apart a key into it's separate components"""

    return (key.split('__') + [None])[:5]

def decimate_points(x, y):
    """Removes unneeded points from a curve"""

    return list(zip(*rdp(list(zip(x, y)))))

def line_dists(points, start, end):
    if np.all(start == end):
        return np.linalg.norm(points - start, axis=1)

    vec = end - start
    cross = np.cross(vec, start - points)
    return np.divide(abs(cross), np.linalg.norm(vec))


def rdp(M, epsilon=0):
    M = np.array(M)
    start, end = M[0], M[-1]
    dists = line_dists(M, start, end)

    index = np.argmax(dists)
    dmax = dists[index]

    if dmax > epsilon:
        result1 = rdp(M[:index + 1], epsilon)
        result2 = rdp(M[index:], epsilon)

        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([start, end])

    return result