"""
Auto ML

Supervised learning using an exhaustive search of ideal pre-processing (if any), algorithms,
and hyper-parameters with feature engineering.
"""

# Dependencies
import os
import csv
import itertools

from dotenv import load_dotenv

from .processors.estimators import ESTIMATOR_NAMES
from .processors.feature_selection import FEATURE_SELECTOR_NAMES
from .processors.scalers import SCALER_NAMES
from .processors.searchers import SEARCHER_NAMES
from .processors.scorers import SCORER_NAMES
from .generalization import generalize
from .model import generate_model
from .import_data import import_data
from .pipeline import generate_pipeline
from .summary import print_summary
from .utils import model_key_to_name

# Load environment variables
load_dotenv()
IGNORE_ESTIMATOR = [x.strip() for x in os.getenv('IGNORE_ESTIMATOR', '').split(',')]
IGNORE_FEATURE_SELECTOR = [x.strip() for x in os.getenv('IGNORE_FEATURE_SELECTOR', '').split(',')]
IGNORE_SCALER = [x.strip() for x in os.getenv('IGNORE_SCALER', '').split(',')]
IGNORE_SEARCHER = [x.strip() for x in os.getenv('IGNORE_SEARCHER', '').split(',')]
IGNORE_SCORER = [x.strip() for x in os.getenv('IGNORE_SCORER', '').split(',')]

def find_best_model(train_set=None, test_set=None, labels=None, label_column=None):
    """Generates all possible models and outputs the generalization results"""

    if train_set is None:
        print('Missing training data')
        return {}

    if test_set is None:
        print('Missing test data')
        return {}

    if label_column is None:
        print('Missing column name for classifier target')
        return {}

    # Import data
    (x_train, y_train, x2, y2, feature_names) = import_data(train_set, test_set, label_column)

    results = []
    total_fits = 0

    all_pipelines = list(itertools.product(
        *[ESTIMATOR_NAMES, FEATURE_SELECTOR_NAMES, SCALER_NAMES, SCORER_NAMES, SEARCHER_NAMES]))

    report = open('report.csv', 'w+')
    reportWriter = csv.writer(report)

    for estimator, feature_selector, scaler, scorer, searcher in all_pipelines:

        # SVM without scaling can loop consume infinite CPU time so
        # prevent that combination here.
        #
        # If any of the steps are matched in the ignore, then continue.
        if (estimator == 'svm' and scaler == 'none') or\
            estimator in IGNORE_ESTIMATOR or\
            feature_selector in IGNORE_FEATURE_SELECTOR or\
            scaler in IGNORE_SCALER or\
            searcher in IGNORE_SEARCHER or\
            scorer in IGNORE_SCORER:
            continue

        key = '__'.join([scaler, feature_selector, estimator, scorer, searcher])
        result = {
            'key': key,
            'scaler': SCALER_NAMES[scaler],
            'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
            'estimator': ESTIMATOR_NAMES[estimator],
            'scorer': SCORER_NAMES[scorer],
            'searcher': SEARCHER_NAMES[searcher]
        }
        print('Generating ' + model_key_to_name(key))

        pipeline = generate_pipeline(scaler, feature_selector, estimator, y_train, scorer, searcher)
        model = generate_model(pipeline[0], feature_names, x_train, y_train, scorer)
        result.update(generalize(model, pipeline[0], x2, y2, labels))

        total_fits += pipeline[1]
        result['selected_features'] = list(model['selected_features'])
        result['best_params'] = model['best_params']

        if not results:
            reportWriter.writerow(result.keys())

        reportWriter.writerow(list([str(i) for i in result.values()]))
        results.append(result)

    report.close()
    print('Total fits generated: %d' % total_fits)
    print_summary(results)
    return results
