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
import gc

from dotenv import load_dotenv
from joblib import dump
from timeit import default_timer as timer
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ml.preprocess import preprocess
from ml.processors.estimators import ESTIMATOR_NAMES
from ml.processors.feature_selection import FEATURE_SELECTOR_NAMES
from ml.processors.scalers import SCALER_NAMES
from ml.processors.searchers import SEARCHER_NAMES
from ml.processors.scorers import SCORER_NAMES
from ml.generalization import generalize
from ml.model import generate_model
from ml.import_data import import_data
from ml.pipeline import generate_pipeline
from ml.precision import precision_recall
from ml.reliability import reliability
from ml.refit import refit_model
from ml.roc import roc
from ml.summary import print_summary
from ml.utils import model_key_to_name
from ml.class_results import save_class_results, generate_ovr_models_and_results
from ml.memory_manager import MemoryManager

# Load environment variables
load_dotenv()

def save_individual_models(main_models, ovr_models, output_path):
    """Save models as individual .joblib files immediately to prevent memory accumulation"""
    
    models_dir = f"{output_path}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save main models as individual files
    if main_models:
        main_models_dir = f"{models_dir}/main_models"
        os.makedirs(main_models_dir, exist_ok=True)
        
        for key, model in main_models.items():
            model_path = f"{main_models_dir}/{key}.joblib"
            dump(model, model_path)
            print(f'Saved main model: {key}.joblib')
    
    # Save OvR models as individual files
    if ovr_models:
        ovr_models_dir = f"{models_dir}/ovr_models"
        os.makedirs(ovr_models_dir, exist_ok=True)
        
        for key, model in ovr_models.items():
            model_path = f"{ovr_models_dir}/{key}.joblib"
            dump(model, model_path)
            print(f'Saved OvR model: {key}.joblib')

def create_final_archives(output_path):
    """Create compressed archives from individual model files at the end (optional)"""
    
    models_dir = f"{output_path}/models"
    main_models_dir = f"{models_dir}/main_models"
    ovr_models_dir = f"{models_dir}/ovr_models"
    
    # Create main models archive if directory exists and has files
    if os.path.exists(main_models_dir) and os.listdir(main_models_dir):
        main_archive_path = f"{models_dir}/main_models.tar.gz"
        print(f'Creating compressed archive: {main_archive_path}')
        
        with tarfile.open(main_archive_path, "w:gz") as tar:
            tar.add(main_models_dir, arcname="main_models")
        
        print(f'Main models archive created: {main_archive_path}')
    
    # Create OvR models archive if directory exists and has files
    if os.path.exists(ovr_models_dir) and os.listdir(ovr_models_dir):
        ovr_archive_path = f"{models_dir}/ovr_models.tar.gz"
        print(f'Creating compressed archive: {ovr_archive_path}')
        
        with tarfile.open(ovr_archive_path, "w:gz") as tar:
            tar.add(ovr_models_dir, arcname="ovr_models")
        
        print(f'OvR models archive created: {ovr_archive_path}')

def calculate_total_pipeline_combinations(ignore_estimator, ignore_scaler, ignore_feature_selector, ignore_searcher):
    """Calculate total number of pipeline combinations without creating the full list"""
    estimators = [x for x in ESTIMATOR_NAMES if x not in ignore_estimator]
    scalers = [x for x in SCALER_NAMES if x not in ignore_scaler]
    feature_selectors = [x for x in FEATURE_SELECTOR_NAMES if x not in ignore_feature_selector]
    searchers = [x for x in SEARCHER_NAMES if x not in ignore_searcher]
    
    return len(estimators) * len(scalers) * len(feature_selectors) * len(searchers)


def create_pipeline_generator(ignore_estimator, ignore_scaler, ignore_feature_selector, ignore_searcher):
    """Create a generator for pipeline combinations to avoid loading all into memory"""
    return itertools.product(*[
        filter(lambda x: False if x in ignore_estimator else True, ESTIMATOR_NAMES),
        filter(lambda x: False if x in ignore_scaler else True, SCALER_NAMES),
        filter(lambda x: False if x in ignore_feature_selector else True, FEATURE_SELECTOR_NAMES),
        filter(lambda x: False if x in ignore_searcher else True, SEARCHER_NAMES),
    ])


def save_model_with_memory_management(memory_mgr, model_dict, output_path, model_type="main"):
    """Save models using memory manager for immediate cleanup"""
    models_dir = f"{output_path}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    if model_type == "main":
        models_subdir = f"{models_dir}/main_models"
    else:
        models_subdir = f"{models_dir}/ovr_models"
    
    os.makedirs(models_subdir, exist_ok=True)
    
    for key, model in model_dict.items():
        model_path = f"{models_subdir}/{key}.joblib"
        memory_mgr.save_model_with_cleanup(model, model_path, key)


def find_best_model(
        train_set=None,
        test_set=None,
        labels=None,
        label_column=None,
        parameters=None,
        output_path='.',
        update_function=lambda x, y: None
    ):
    """Generates all possible models and outputs the generalization results with memory management"""

    # Initialize memory manager
    memory_mgr = MemoryManager(warning_threshold=0.75, critical_threshold=0.85)
    memory_mgr.log_memory_stats("initialization")

    start = timer()

    # Parse parameters
    ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
    ignore_feature_selector = \
        [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
    ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
    ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
    shuffle = False if parameters.get('ignore_shuffle', '') != '' else True
    scorers = [x for x in SCORER_NAMES if x not in \
        [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]]

    # Validation
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
    
    memory_mgr.log_memory_stats("after_data_import")

    # Initialize tracking variables
    total_fits = {}
    csv_header_written = False
    reoptimize_ovr = parameters.get('reoptimize_ovr', 'false').lower() == 'true'
    main_models_count = 0
    ovr_models_count = 0
    n_classes = len(np.unique(y_train))

    # Create pipeline generator instead of full list
    pipeline_generator = create_pipeline_generator(
        ignore_estimator, ignore_scaler, ignore_feature_selector, ignore_searcher
    )
    total_pipeline_count = calculate_total_pipeline_combinations(
        ignore_estimator, ignore_scaler, ignore_feature_selector, ignore_searcher
    )

    if total_pipeline_count == 0:
        print('No pipelines to run with the current configuration')
        return False

    print(f"Processing {total_pipeline_count} pipeline combinations with memory management")

    # Open report files
    report = open(output_path + '/report.csv', 'w+')
    report_writer = csv.writer(report)

    performance_report = open(output_path + '/performance_report.csv', 'w+')
    performance_report_writer = csv.writer(performance_report)
    performance_report_writer.writerow(['key', 'train_time (s)'])

    # Process pipelines one at a time with memory management
    for index, (estimator, scaler, feature_selector, searcher) in enumerate(pipeline_generator):
        
        # Check memory status before processing each pipeline
        should_continue, reason = memory_mgr.should_continue_processing()
        if not should_continue:
            print(f"⚠️  Stopping processing early: {reason}")
            print(f"Processed {index}/{total_pipeline_count} pipelines before stopping")
            break

        # Trigger callback for task monitoring
        update_function(index, total_pipeline_count)

        key = '__'.join([scaler, feature_selector, estimator, searcher])
        print(f'Generating {model_key_to_name(key)} ({index+1}/{total_pipeline_count})')
        
        memory_mgr.log_memory_stats("before_pipeline", key)

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

        memory_mgr.log_memory_stats("after_model_generation", key)

        # Process each scorer
        for scorer in scorers:
            scorer_key = key + '__' + scorer
            candidates = refit_model(pipeline[0], model['features'], estimator, scorer, x_train, y_train)
            total_fits[estimator] += len(candidates)

            # Process each candidate model
            for position, candidate in enumerate(candidates):
                
                # Check memory before processing each candidate
                memory_status = memory_mgr.check_memory_status()
                if memory_status == 'critical':
                    print(f"⚠️  Critical memory level reached, performing emergency cleanup")
                    if not memory_mgr.emergency_cleanup():
                        print(f"❌ Emergency cleanup failed, skipping remaining candidates")
                        break

                candidate_key = scorer_key + '__' + str(position)
                
                # Build result dictionary
                result = {
                    'key': candidate_key,
                    'class_type': 'multiclass' if n_classes > 2 else 'binary',
                    'class_index': None,
                    'scaler': SCALER_NAMES[scaler],
                    'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
                    'algorithm': ESTIMATOR_NAMES[estimator],
                    'searcher': SEARCHER_NAMES[searcher],
                    'scorer': SCORER_NAMES[scorer],
                }

                print(f'\t#{position+1}')
                
                # Compute metrics
                result.update(generalize(pipeline[0], model['features'], candidate['best_estimator'], x2, y2, labels))
                result.update({
                    'selected_features': list(model['selected_features']),
                    'feature_scores': model['feature_scores'],
                    'best_params': candidate['best_params']
                })
                
                # ROC metrics
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
                
                # Additional metrics
                result.update(reliability(pipeline[0], model['features'], candidate['best_estimator'], x2, y2))
                result.update(precision_recall(pipeline[0], model['features'], candidate['best_estimator'], x2, y2))
                
                # Write CSV header if first result
                if not csv_header_written:
                    report_writer.writerow(result.keys())
                    csv_header_written = True

                report_writer.writerow(list([str(i) for i in result.values()]))

                # Save main model with memory management
                single_main_model = {result['key']: candidate['best_estimator']}
                save_model_with_memory_management(memory_mgr, single_main_model, output_path, "main")
                main_models_count += 1

                # Handle multiclass OvR models
                new_ovr_models = {}
                csv_entries = []
                class_data = {}
                
                if n_classes > 2:
                    # Generate OvR models and results
                    csv_entries, class_data, new_ovr_models, additional_fits = generate_ovr_models_and_results(
                        pipeline[0], model['features'], candidate['best_estimator'], result,
                        x_train, y_train, x_test, y_test, x2, y2, labels,
                        estimator, scorer, reoptimize_ovr
                    )
                    
                    total_fits[estimator] += additional_fits
                    
                    # Save OvR models with memory management
                    if new_ovr_models:
                        save_model_with_memory_management(memory_mgr, new_ovr_models, output_path, "ovr")
                        ovr_models_count += len(new_ovr_models)
                    
                    # Write CSV entries for OvR models
                    for csv_entry in csv_entries:
                        report_writer.writerow(list([str(i) for i in csv_entry.values()]))
                    
                    # Save class-specific data
                    class_results_dir = output_path + '/class_results'
                    save_class_results(class_data, class_results_dir, result['key'])

                # AGGRESSIVE CLEANUP after each model
                memory_mgr.cleanup_model_iteration(
                    pipeline=pipeline,
                    model=model,
                    candidate=candidate,
                    result=result,
                    single_main_model=single_main_model,
                    new_ovr_models=new_ovr_models,
                    csv_entries=csv_entries,
                    class_data=class_data,
                    roc_auc=roc_auc
                )
                
                memory_mgr.log_memory_stats("after_model_cleanup", candidate_key)

        # Additional cleanup after each pipeline
        memory_mgr.aggressive_cleanup()

    # Final processing
    train_time = timer() - start
    print(f'\tTotal run time is {train_time:.4f} seconds\n')
    performance_report_writer.writerow(['total', train_time])
   
    report.close()
    performance_report.close()
    
    print('Total fits generated', sum(total_fits.values()))
    print_summary(output_path + '/report.csv')
     
    print(f'All models saved with memory management: {main_models_count} main models, {ovr_models_count} OvR models')
    
    # Create compressed archives
    create_final_archives(output_path)

    # Update metadata
    metadata.update({
        'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'fits': total_fits,
        'ovr_reoptimized': reoptimize_ovr,
        'ovr_models_count': ovr_models_count,
        'main_models_count': main_models_count,
        'n_classes': int(n_classes),  # Ensure native Python int
        'memory_management': memory_mgr.get_memory_summary()
    })

    # Save metadata
    if output_path != '.':
        metadata_path = output_path + '/metadata.json'
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
