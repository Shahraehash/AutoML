"""
Auto ML

Supervised learning using an exhaustive search of ideal pre-processing (if any), algorithms,
and hyper-parameters with feature engineering.
"""

# Dependencies
import os
import csv
import json
import itertools

from dotenv import load_dotenv
from sklearn.utils import parallel_backend

from .processors.estimators import ESTIMATOR_NAMES
from .processors.feature_selection import FEATURE_SELECTOR_NAMES
from .processors.scalers import SCALER_NAMES
from .processors.searchers import SEARCHER_NAMES
from .processors.scorers import SCORER_NAMES
from .generalization import generalize
from .model import generate_model
from .import_data import import_data
from .pipeline import generate_pipeline
from .reliability import reliability
from .refit import refit_model
from .roc import roc
from .summary import print_summary
from .utils import model_key_to_name

# Load environment variables
load_dotenv()

def find_best_model(train_set=None, test_set=None, labels=None, label_column=None, output_path='.'):
    """Generates all possible models and outputs the generalization results"""

    ignore_estimator = [x.strip() for x in os.getenv('IGNORE_ESTIMATOR', '').split(',')]
    ignore_feature_selector = \
        [x.strip() for x in os.getenv('IGNORE_FEATURE_SELECTOR', '').split(',')]
    ignore_scaler = [x.strip() for x in os.getenv('IGNORE_SCALER', '').split(',')]
    ignore_searcher = [x.strip() for x in os.getenv('IGNORE_SEARCHER', '').split(',')]
    scorers = [x for x in SCORER_NAMES if x not in \
        [x.strip() for x in os.getenv('IGNORE_SCORER', '').split(',')]]

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
    (x_train, x_test, y_train, y_test, x2, y2, feature_names, metadata) = \
        import_data(train_set, test_set, label_column)

    results = []
    total_fits = {}

    all_pipelines = list(itertools.product(
        *[ESTIMATOR_NAMES, FEATURE_SELECTOR_NAMES, SCALER_NAMES, SEARCHER_NAMES]))

    report = open(output_path + '/report.csv', 'w+')
    reportWriter = csv.writer(report)

    for estimator, feature_selector, scaler, searcher in all_pipelines:

        # SVM without scaling can loop consuming infinite CPU time so
        # we prevent that combination here.
        #
        # If any of the steps are matched in the ignore, then continue.
        if (estimator == 'svm' and scaler == 'none') or\
            estimator in ignore_estimator or\
            feature_selector in ignore_feature_selector or\
            scaler in ignore_scaler or\
            searcher in ignore_searcher:
            continue

        key = '__'.join([scaler, feature_selector, estimator, searcher])
        roc_curves = {}
        print('Generating ' + model_key_to_name(key))

        # Generate the pipeline
        pipeline = \
            generate_pipeline(scaler, feature_selector, estimator, y_train, scorers, searcher)

        if not estimator in total_fits:
            total_fits[estimator] = 0
        total_fits[estimator] += pipeline[1]

        # Fit the pipeline
        with parallel_backend('threading'):
            model = generate_model(pipeline[0], feature_names, x_train, y_train)

        if pipeline[2]:
            roc_curves = pipeline[2].get_mean()

        for scorer in scorers:
            key += '__' + scorer
            model.update(
                refit_model(pipeline[0], model['features'], estimator, scorer, x_train, y_train))

            total_fits[estimator] += 1

            result = {
                'key': key,
                'scaler': SCALER_NAMES[scaler],
                'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
                'estimator': ESTIMATOR_NAMES[estimator],
                'searcher': SEARCHER_NAMES[searcher],
                'scorer': SCORER_NAMES[scorer]
            }

            result.update(generalize(model, pipeline[0], x2, y2, labels))
            result.update({
                'selected_features': list(model['selected_features']),
                'best_params': model['best_params']
            })
            result.update(roc_curves)
            result.update(roc(pipeline[0], model, x_test, y_test, 'test'))
            result.update(roc(pipeline[0], model, x2, y2, 'generalization'))
            result.update(reliability(pipeline[0], model, x2, y2))

            if not results:
                reportWriter.writerow(result.keys())

            reportWriter.writerow(list([str(i) for i in result.values()]))
            results.append(result)

    report.close()
    print('Total fits generated', sum(total_fits.values()))
    print_summary(results)

    # Update the metadata and write it out
    metadata.update({'fits': total_fits})
    with open(output_path + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

    return results
