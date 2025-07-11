"""
One-vs-Rest (OvR) Classifier

This module provides the OvRClassifier class for handling One-vs-Rest classification tasks.
It extends the AutoMLClassifier base class with OvR-specific logic, creating separate
binary classifiers for each class.
"""

import json
import numpy as np
import pandas as pd
import os
import csv
import time
import itertools
import gzip
import pickle
from timeit import default_timer as timer
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy import interpolate
from nyoka import skl_to_pmml, xgboost_to_pmml

from .multiclass_macro_classifier import MulticlassMacroClassifier
from .utils.import_data import import_data
from .utils.preprocess import preprocess
from .utils.utils import model_key_to_name, decimate_points, explode_key
from .utils.stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci
from .utils.summary import print_summary
from .processors.estimators import ESTIMATORS, ESTIMATOR_NAMES, get_xgb_classifier
from .processors.scorers import SCORER_NAMES
from .processors.scalers import SCALERS, SCALER_NAMES
from .processors.feature_selection import FEATURE_SELECTORS, FEATURE_SELECTOR_NAMES
from .processors.searchers import SEARCHER_NAMES, SEARCHERS 
from .processors.debug import Debug

class OvRClassifier(MulticlassMacroClassifier):
    """
    Handles One-vs-Rest classification tasks.
    
    This classifier creates separate binary classifiers for each class in a multiclass
    problem, treating each class against all others. It can optionally re-optimize
    each OvR model or use the main multiclass model for efficiency.
    """
    
    def __init__(self, parameters, output_path='.', update_function=lambda x, y: None):
        """
        Initialize the OvR Classifier.
        
        Args:
            parameters (dict): Configuration parameters for the classifier
            output_path (str): Path where results and models will be saved
            update_function (callable): Callback function for progress updates
        """
        super().__init__(parameters, output_path, update_function)
        
        # OvR-specific parameter
        self.reoptimize_ovr = parameters.get('reoptimize_ovr', 'false').lower() == 'true'
    
    def compute_binary_class_results(self, pipeline, features, estimator, x_val, y_val, x_train=None, y_train=None, x_test=None, y_test=None):
        """Compute metrics for binary classification (used for OvR models in re-optimization mode)"""
        
        # Compute reliability, precision_recall, and roc for binary classification
        generalization_data = self.generalize(pipeline, features, estimator, x_val, y_val)
        reliability_data = self.reliability(pipeline, features, estimator, x_val, y_val)
        precision_data = self.precision_recall(pipeline, features, estimator, x_val, y_val)
        roc_data = self.roc(pipeline, features, estimator, x_val, y_val)
        
        # Compute training ROC AUC if training data is provided
        training_roc_auc = None
        roc_delta = None
        
        if x_train is not None and y_train is not None:
            training_roc_data = self.roc(pipeline, features, estimator, x_train, y_train)
            training_roc_auc = training_roc_data['roc_auc']
            
            # Calculate ROC delta
            if roc_data['roc_auc'] is not None and training_roc_auc is not None:
                roc_delta = round(abs(roc_data['roc_auc'] - training_roc_auc), 4)
        
        # Compute test ROC metrics if test data is provided
        test_roc_data = None
        if x_test is not None and y_test is not None:
            test_roc_data = self.roc(pipeline, features, estimator, x_test, y_test)
        
        return {
            'generalization': generalization_data,
            'reliability': reliability_data,
            'precision_recall': precision_data,
            'roc_auc': roc_data,
            'training_roc_auc': training_roc_auc,
            'roc_delta': roc_delta,
            'test_roc_data': test_roc_data
        }

    def generate_ovr_models_and_results(self, pipeline, features, main_model, main_result, x_train, y_train, x_val, y_val, x_test, y_test, labels, estimator, scorer):
        """Generate OvR models and return both CSV entries and class data for .pkl.gz storage"""
        n_classes = len(np.unique(y_train))
        unique_classes = sorted(np.unique(y_train))
        
        csv_entries = []
        ovr_models = {}
        total_fits = 0
        
        # Storage for class-specific data (for .pkl.gz file)
        all_class_data = {
            'model_key': main_result['key'],
            'n_classes': n_classes,
            'class_data': {}
        }
        
        for class_idx in range(n_classes):
            # Create binary labels for this class vs rest
            actual_class_value = unique_classes[class_idx]
            y_binary = (y_train == actual_class_value).astype(int)
            y_val_binary = (y_val == actual_class_value).astype(int)
            y_test_binary = (y_test == actual_class_value).astype(int)
        
            # Re-optimization mode: Train actual OvR model
            ovr_candidates = self.refit_model(
                pipeline, features, estimator, scorer, x_train, y_binary
            )
            
            ovr_model = ovr_candidates[0]['best_estimator']
            ovr_best_params = ovr_candidates[0]['best_params']
            total_fits += len(ovr_candidates)
            
            # Store OvR model
            ovr_key = f"{main_result['key']}_ovr_class_{class_idx}"
            ovr_models[ovr_key] = ovr_model

            # Re-optimization mode: Use binary classification path
            # OvR model was trained on binary data, so evaluate it on binary data
            class_metrics = self.compute_binary_class_results(pipeline, features, ovr_model, x_val, y_val_binary, x_train, y_binary, x_test, y_test_binary)
                
            # Store class data for .pkl.gz file
            all_class_data['class_data'][class_idx] = class_metrics
            
            # Create CSV entry for this OvR model using already computed metrics
            csv_entry = create_ovr_csv_entry(
                main_result, class_idx, ovr_best_params, class_metrics
            )
            
            csv_entries.append(csv_entry)
        
        return csv_entries, all_class_data, ovr_models, total_fits

    def find_best_model(self, train_set=None, test_set=None, labels=None, label_column=None, parameters=None, output_path='.', update_function=lambda x, y: None):
        """
        Multiclass classification with macro-averaging and class-specific results (no model re-optimization).
        """
        start = timer()     
        
        ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
        ignore_feature_selector = \
            [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
        ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
        ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
        shuffle = False if parameters.get('ignore_shuffle', '') != '' else True
        scorers = [x for x in SCORER_NAMES if x not in \
            [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]]

        # Basic validation
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
        (x_train, x_val, y_train, y_val, x_test, y_test, feature_names, metadata) = \
            import_data(train_set, test_set, label_column)

        # Determine number of classes - must be multiclass
        n_classes = len(np.unique(y_train))
        print(f'Detected {n_classes} classes in target column')
        
        if n_classes <= 2:
            raise ValueError(f"MulticlassMacroClassifier expects more than 2 classes, got {n_classes}")

        total_fits = {}
        csv_header_written = False

        # Memory-efficient model storage
        main_models = {}  # Store main models
        ovr_models = {}   # Store references to main models (no separate training)

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

        print(f"Starting multiclass macro-averaging classification with {len(all_pipelines)} pipeline combinations...")
        print(f"Number of classes: {n_classes}")

        for index, (estimator, scaler, feature_selector, searcher) in enumerate(all_pipelines):

            # Trigger callback for task monitoring
            update_function(index, len(all_pipelines))

            key = '__'.join([scaler, feature_selector, estimator, searcher])
            print('Generating ' + model_key_to_name(key))

            # Generate the pipeline
            pipeline = self.generate_pipeline(scaler, feature_selector, estimator, y_train, scorers, searcher, shuffle, custom_hyper_parameters)

            if not estimator in total_fits:
                total_fits[estimator] = 0
            total_fits[estimator] += pipeline[1]

            # Fit the pipeline
            model = self.generate_model(pipeline[0], feature_names, x_train, y_train)
            performance_report_writer.writerow([key, model['train_time']])

            for scorer in scorers:
                scorer_key = key + '__' + scorer
                candidates = self.refit_model(pipeline[0], model['features'], estimator, scorer, x_train, y_train)
                total_fits[estimator] += len(candidates)

                for position, candidate in enumerate(candidates):
                    print('\t#%d' % (position+1))
                    
                    # Create base result
                    result = {
                        'key': scorer_key + '__' + str(position),
                        'class_type': 'multiclass',
                        'class_index': None,  # Main model has no specific class
                        'scaler': SCALER_NAMES[scaler],
                        'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
                        'algorithm': ESTIMATOR_NAMES[estimator],
                        'searcher': SEARCHER_NAMES[searcher],
                        'scorer': SCORER_NAMES[scorer],
                    }

                    # Store main model
                    main_models[result['key']] = candidate['best_estimator']

                    # Update result with evaluation metrics
                    result.update(self.generalize(pipeline[0], model['features'], candidate['best_estimator'], x_val, y_val, labels))
                    result.update({
                        'selected_features': list(model['selected_features']),
                        'feature_scores': model['feature_scores'],
                        'best_params': candidate['best_params']
                    })

                    roc_auc = self.roc(pipeline[0], model['features'], candidate['best_estimator'], x_val, y_val)
                    result.update({
                        'test_fpr': roc_auc['fpr'],
                        'test_tpr': roc_auc['tpr'],
                        'training_roc_auc': roc_auc['roc_auc']
                    })
                    result['roc_delta'] = round(abs(result['roc_auc'] - result['training_roc_auc']), 4)
                    roc_auc = self.roc(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test)
                    result.update({
                        'generalization_fpr': roc_auc['fpr'],
                        'generalization_tpr': roc_auc['tpr']
                    })
                    result.update(self.reliability(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test))
                    result.update(self.precision_recall(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test))
                    
                    print(f"Full Result: {result}")

                    # Write main model result
                    if not csv_header_written:
                        report_writer.writerow(result.keys())
                        csv_header_written = True

                    report_writer.writerow(list([str(i) for i in result.values()]))

                    # Generate OvR metrics for each class (no retraining - macro mode)
                    print(f"\t\tGenerating OvR metrics for {n_classes} classes...")
                    
                    for class_idx, class_label in enumerate(labels):
                        # Use the simplified function to handle all OvR logic
                        csv_entries, class_data, new_ovr_models, additional_fits = self.generate_ovr_models_and_results(
                            pipeline[0], model['features'], candidate['best_estimator'], result,
                            x_train, y_train, x_val, y_val, x_test, y_test, labels,
                            estimator, scorer
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
                        self.save_class_results(class_data, class_results_dir, result['key'])

                        print(f"\t\tGenerated {n_classes} OvR metric entries")

        train_time = timer() - start
        print('\tTotal run time is {:.4f} seconds'.format(train_time), '\n')
        performance_report_writer.writerow(['total', train_time])

        report.close()
        performance_report.close()
        print('Total fits generated', sum(total_fits.values()))
        print_summary(output_path + '/report.csv')
        
        self.save_model_archives(main_models, ovr_models, output_path)

        # Update the metadata and write it out
        metadata.update({
            'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'fits': total_fits,
            'ovr_reoptimized': False,
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


    @staticmethod
    def create_model(key, hyper_parameters, selected_features, dataset_path=None, label_column=None, output_path='.', threshold=.5):
        """Refits the requested model and pickles it for export"""

        if dataset_path is None:
            print('Missing dataset path')
            return {}

        if label_column is None:
            print('Missing column name for classifier target')
            return {}

        # Import data
        (x_train, _, y_train, _, x_test, y_test, features, _) = \
            import_data(dataset_path + '/train.csv', dataset_path + '/test.csv', label_column)

        # Get pipeline details from the key
        scaler, feature_selector, estimator, _, _ = explode_key(key)
        steps = []

        # Drop the unused features
        if 'pca-' not in feature_selector:
            for index, feature in reversed(list(enumerate(features))):
                if feature not in selected_features:
                    x_train = np.delete(x_train, index, axis=1)
                    x_test = np.delete(x_test, index, axis=1)

        # Add the scaler, if used
        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        # Add the feature transformer
        if 'pca-' in feature_selector:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        # Add the estimator with proper XGBoost configuration
        if estimator == 'gb':
            n_classes = len(pd.Series(y_train).unique())
            base_estimator = get_xgb_classifier(n_classes)
        else:
            base_estimator = ESTIMATORS[estimator]
        
        steps.append(('estimator', base_estimator.set_params(**hyper_parameters)))

        # Fit the pipeline using the same training data
        pipeline = Pipeline(steps)
        model = self.train_model(pipeline, selected_features, x_train, y_train)

        # If the model is DNN or RF, attempt to swap the estimator for a pickled one
        if os.path.exists(output_path + '/models/' + key + '.joblib'):
            pickled_estimator = load(output_path + '/models/' + key + '.joblib')
            pipeline = Pipeline(pipeline.steps[:-1] + [('estimator', pickled_estimator)])

        unique_labels = sorted(y_test.unique())
        labels = [f'Class {int(cls)}' for cls in unique_labels]

        # Assess the model performance and store the results
        generalization_result = self.evaluate_generalization(pipeline, model['features'], pipeline['estimator'], x_test, y_test, labels)
        with open(output_path + '/pipeline.json', 'w') as statsfile:
            json.dump(generalization_result, statsfile)

        # Dump the pipeline to a file
        dump(pipeline, output_path + '/pipeline.joblib')
        pd.DataFrame([selected_features]).to_csv(output_path + '/input.csv', index=False, header=False)

        # Export the model as a PMML
        try:
            if estimator == 'gb':
                xgboost_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
            else:
                skl_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
        except Exception:
            try:
                os.remove(output_path + '/pipeline.pmml')
            except OSError:
                pass

        return generalization_result

    @staticmethod
    def generalize_ensemble(total_models, job_folder, dataset_folder, label):
        x_test, y_test, feature_names, _, _ = import_csv(dataset_folder + '/test.csv', label)

        data = pd.DataFrame(x_test, columns=feature_names)

        soft_result = self.predict_ensemble(total_models, data, job_folder, 'soft')
        hard_result = self.predict_ensemble(total_models, data, job_folder, 'hard')

        unique_labels = sorted(y_test.unique())
        labels = [f'Class {int(cls)}' for cls in unique_labels]

        return {
            'soft_generalization': self.generalization_report(labels, y_test, soft_result['predicted'], soft_result['probability']),
            'hard_generalization': self.generalization_report(labels, y_test, hard_result['predicted'], hard_result['probability'])
        }

    @staticmethod
    def additional_precision(payload, label, folder, class_index=None):
        """Return additional precision recall curve"""

        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.precision_recall(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)
    
    @staticmethod
    def additional_reliability(payload, label, folder, class_index=None):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.reliability(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)

    @staticmethod
    def additional_roc(payload, label, folder, class_index=None):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.roc(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)


    @staticmethod
    def predict(data, path='.', threshold=.5):
        """Predicts against the provided data"""

        # Load the pipeline
        pipeline = load(path + '.joblib')

        data = pd.DataFrame(data).dropna().values

        probability = pipeline.predict_proba(data)
        if threshold == .5:
            predicted = pipeline.predict(data)
        else:
            predicted = (probability[:, 1] >= threshold).astype(int)

        return {
            'predicted': predicted.tolist(),
            'probability': [sublist[predicted[index]] for index, sublist in enumerate(probability.tolist())]
        }

    @staticmethod
    def predict_ensemble(total_models, data, path='.', vote_type='soft'):
        """Predicts against the provided data by creating an ensemble of the selected models"""

        probabilities = []
        predictions = []

        for x in range(total_models):
            pipeline = load(path + '/ensemble' + str(x) + '.joblib')

            with open(path + '/ensemble' + str(x) + '_features.json') as feature_file:
                features = json.load(feature_file)

            selected_data = data[features].dropna().to_numpy()
            probabilities.append(pipeline.predict_proba(selected_data))
            predictions.append(pipeline.predict(selected_data))

        predictions = np.asarray(predictions).T
        probabilities = np.average(np.asarray(probabilities), axis=0)

        if vote_type == 'soft':
            predicted = np.argmax(probabilities, axis=1)
        else:
            predicted = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions.astype('int')
            )

        return {
            'predicted': predicted.tolist(),
            'probability': [sublist[predicted[index]] for index, sublist in enumerate(probabilities.tolist())]
        }
