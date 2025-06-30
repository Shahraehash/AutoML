"""
Auto ML

Supervised learning using an exhaustive search of ideal pre-processing (if any), algorithms,
and hyper-parameters with feature engineering.
"""

import csv
import json
import time
import itertools
import numpy as np
import tarfile
import tempfile
import os

from dotenv import load_dotenv
from joblib import dump
from timeit import default_timer as timer
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .processors.estimators import ESTIMATOR_NAMES
from .processors.feature_selection import FEATURE_SELECTOR_NAMES
from .processors.scalers import SCALER_NAMES
from .processors.searchers import SEARCHER_NAMES
from .processors.scorers import SCORER_NAMES
from .generalization import generalize
from .model import generate_model
from .import_data import import_data
from .pipeline import generate_pipeline
from .precision import precision_recall
from .reliability import reliability
from .refit import refit_model
from .roc import roc
from .summary import print_summary
from .utils import model_key_to_name
from .class_results import save_class_results, generate_ovr_models_and_results

# Load environment variables
load_dotenv()

def save_model_archives(main_models, ovr_models, output_path):
    """Save all models in compressed archives"""
    
    models_dir = f"{output_path}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f'Saving {len(main_models)} main models to compressed archive...')
    
    # Save main models archive
    if main_models:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save individual models to temp directory
            for key, model in main_models.items():
                temp_path = f"{temp_dir}/{key}.joblib"
                dump(model, temp_path)
            
            # Create compressed archive
            with tarfile.open(f"{models_dir}/main_models.tar.gz", "w:gz") as tar:
                tar.add(temp_dir, arcname="main_models")
        
        print(f'Main models archive saved: {models_dir}/main_models.tar.gz')
    
    # Save OvR models archive (if any)
    if ovr_models:
        print(f'Saving {len(ovr_models)} OvR models to compressed archive...')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for key, model in ovr_models.items():
                temp_path = f"{temp_dir}/{key}.joblib"
                dump(model, temp_path)
            
            with tarfile.open(f"{models_dir}/ovr_models.tar.gz", "w:gz") as tar:
                tar.add(temp_dir, arcname="ovr_models")
        
        print(f'OvR models archive saved: {models_dir}/ovr_models.tar.gz')

def find_best_model(
        train_set=None,
        test_set=None,
        labels=None,
        label_column=None,
        parameters=None,
        output_path='.',
        update_function=lambda x, y: None
    ):
    """Generates all possible models and outputs the generalization results"""

    start = timer()

    ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
    ignore_feature_selector = \
        [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
    ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
    ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
    shuffle = False if parameters.get('ignore_shuffle', '') != '' else True
    scorers = [x for x in SCORER_NAMES if x not in \
        [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]]

    if train_set is None:
        print('Missing training data')
        return {}

    if test_set is None:
        print('Missing test data')
        return {}

    if label_column is None:
        print('Missing column name for classifier target')
        return {}

    custom_hyper_parameters = json.loads(parameters['hyper_parameters'])\
        if 'hyper_parameters' in parameters else None

    # Import data
    (x_train, x_test, y_train, y_test, x2, y2, feature_names, metadata) = \
        import_data(train_set, test_set, label_column)
    
    

    total_fits = {}
    csv_header_written = False
    reoptimize_ovr = False  # Initialize this variable
    
    # Memory-efficient model storage
    main_models = {}  # Store main models in memory
    ovr_models = {}   # Store OvR models in memory

    all_pipelines = list(itertools.product(*[
        filter(lambda x: False if x in ignore_estimator else True, ESTIMATOR_NAMES),
        filter(lambda x: False if x in ignore_scaler else True, SCALER_NAMES),
        filter(lambda x: False if x in ignore_feature_selector else True, FEATURE_SELECTOR_NAMES),
        filter(lambda x: False if x in ignore_searcher else True, SEARCHER_NAMES),
    ]))

    if not len(all_pipelines):
        print('No pipelines to run with the current configuration')
        return False

    report = open(output_path + '/report.csv', 'w+')
    report_writer = csv.writer(report)

    performance_report = open(output_path + '/performance_report.csv', 'w+')
    performance_report_writer = csv.writer(performance_report)
    performance_report_writer.writerow(['key', 'train_time (s)'])

    for index, (estimator, scaler, feature_selector, searcher) in enumerate(all_pipelines):

        # Trigger a callback for task monitoring purposes
        update_function(index, len(all_pipelines))

        key = '__'.join([scaler, feature_selector, estimator, searcher])
        print('Generating ' + model_key_to_name(key))

        # Generate the pipeline
        pipeline = generate_pipeline(
            scaler,
            feature_selector,
            estimator,
            y_train,
            scorers,
            searcher,
            shuffle,
            custom_hyper_parameters
        )

        if not estimator in total_fits:
            total_fits[estimator] = 0
        total_fits[estimator] += pipeline[1]

        # Fit the pipeline
        model = generate_model(pipeline[0], feature_names, x_train, y_train)
        performance_report_writer.writerow([key, model['train_time']])

        for scorer in scorers:
            key += '__' + scorer
            candidates = refit_model(pipeline[0], model['features'], estimator, scorer, x_train, y_train)
            total_fits[estimator] += len(candidates)

            for position, candidate in enumerate(candidates):
                # Check if this is a multiclass problem
                n_classes = len(np.unique(y_train))
                result = {
                    'key': key + '__' + str(position),
                    'class_type': 'multiclass' if n_classes > 2 else 'binary',
                    'class_index': None,
                    'scaler': SCALER_NAMES[scaler],
                    'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
                    'algorithm': ESTIMATOR_NAMES[estimator],
                    'searcher': SEARCHER_NAMES[searcher],
                    'scorer': SCORER_NAMES[scorer],
                }

                print('\t#%d' % (position+1))
                # Store main model in memory instead of saving immediately
                main_models[result['key']] = candidate['best_estimator']

                result.update(generalize(pipeline[0], model['features'], candidate['best_estimator'], x2, y2, labels))
                result.update({
                    'selected_features': list(model['selected_features']),
                    'feature_scores': model['feature_scores'],
                    'best_params': candidate['best_params']
                })
                roc_auc = roc(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test)
                result.update({
                  'test_fpr': roc_auc['fpr'],
                  'test_tpr': roc_auc['tpr'],
                  'training_roc_auc': roc_auc['roc_auc']
                })
                result['roc_delta'] = round(abs(result['roc_auc'] - result['training_roc_auc']), 4)
                roc_auc = roc(pipeline[0], model['features'], candidate['best_estimator'], x2, y2)
                result.update({
                  'generalization_fpr': roc_auc['fpr'],
                  'generalization_tpr': roc_auc['tpr']
                })
                result.update(reliability(pipeline[0], model['features'], candidate['best_estimator'], x2, y2))
                result.update(precision_recall(pipeline[0], model['features'], candidate['best_estimator'], x2, y2))

                # Write main model results first
                if not csv_header_written:
                    report_writer.writerow(result.keys())
                    csv_header_written = True

                report_writer.writerow(list([str(i) for i in result.values()]))

                # Generate and save OvR binary models for multiclass problems
                if n_classes > 2:
                    # Extract OvR optimization setting
                    reoptimize_ovr = parameters.get('reoptimize_ovr', 'false').lower() == 'true'
                    # Use the simplified function to handle all OvR logic
                    csv_entries, class_data, new_ovr_models, additional_fits = generate_ovr_models_and_results(
                        pipeline[0], model['features'], candidate['best_estimator'], result,
                        x_train, y_train, x_test, y_test, x2, y2, labels,
                        estimator, scorer, reoptimize_ovr
                    )
                    
                    # Update total fits
                    total_fits[estimator] += additional_fits
                    
                    # Store any new OvR models
                    ovr_models.update(new_ovr_models)
                    
                    # Write all CSV entries
                    for csv_entry in csv_entries:
                        report_writer.writerow(list([str(i) for i in csv_entry.values()]))
                    
                    # Save class-specific data as .pkl.gz file
                    class_results_dir = output_path + '/class_results'
                    save_class_results(class_data, class_results_dir, result['key'])
                            
                    

    train_time = timer() - start
    print('\tTotal run time is {:.4f} seconds'.format(train_time), '\n')
    performance_report_writer.writerow(['total', train_time])
   
    report.close()
    performance_report.close()
    print('Total fits generated', sum(total_fits.values()))
    print_summary(output_path + '/report.csv')
     
    save_model_archives(main_models, ovr_models, output_path)

    # Update the metadata and write it out
    metadata.update({
        'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'fits': total_fits,
        'ovr_reoptimized': reoptimize_ovr,
        'ovr_models_count': len(ovr_models),
        'main_models_count': len(main_models),
        'n_classes': n_classes
    })
        

    if output_path != '.':
        
        metadata_path = output_path + '/metadata.json'
        # Check if metadata file exists and load existing data
        existing_metadata = {}
        if os.path.exists(metadata_path):
   
            with open(metadata_path, 'r') as metafile:
                existing_metadata = json.load(metafile)
        
            # Update with new metadata
            existing_metadata.update(metadata)
            
            # Write the updated metadata
            with open(metadata_path, 'w') as metafile:
                json.dump(existing_metadata, metafile, indent=2)
        
    return True
