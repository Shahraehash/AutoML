"""
Base AutoML Classifier

This module provides the base class for all AutoML classification tasks.
It contains common functionality shared across binary, multiclass, and OvR classifiers.
"""

import csv
import json
import time
import itertools
import numpy as np
import tarfile
import tempfile
import os
import pickle
import gzip
import shutil
from timeit import default_timer as timer
from joblib import dump, load
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score,\
    confusion_matrix, classification_report, f1_score, roc_curve,\
    matthews_corrcoef
    
from ..processors.estimators import ESTIMATOR_NAMES, ESTIMATORS, get_xgb_classifier
from ..processors.feature_selection import FEATURE_SELECTOR_NAMES, FEATURE_SELECTORS
from ..processors.scalers import SCALER_NAMES, SCALERS
from ..processors.searchers import SEARCHER_NAMES, SEARCHERS
from ..processors.scorers import SCORER_NAMES
from ..processors.debug import Debug
from ..preprocess import preprocess
from ..utils import model_key_to_name
from ..stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci


class AutoMLClassifier:
    """
    Master base class for all AutoML classification tasks.
    
    This class provides common functionality for:
    - Pipeline generation and management
    - Model training and evaluation
    - Result storage and archiving
    - Progress tracking and reporting
    """
    
    def __init__(self, parameters, output_path='.', update_function=lambda x, y: None):
        """
        Initialize the AutoML classifier.
        
        Args:
            parameters (dict): Configuration parameters for the classifier
            output_path (str): Path where results and models will be saved
            update_function (callable): Callback function for progress updates
        """
        self.parameters = parameters
        self.output_path = output_path
        self.update_function = update_function
        
        # Storage for models and results
        self.main_models = {}
        self.ovr_models = {}
        self.results = []
        self.total_fits = {}
        
        # Parse parameters
        self.ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
        self.ignore_feature_selector = [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
        self.ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
        self.ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
        self.shuffle = False if parameters.get('ignore_shuffle', '') != '' else True
        self.scorers = [x for x in SCORER_NAMES if x not in 
                       [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]]
        
        self.custom_hyper_parameters = json.loads(parameters['hyper_parameters']) \
            if 'hyper_parameters' in parameters else None
        
        # Initialize report files
        self.report = None
        self.report_writer = None
        self.performance_report = None
        self.performance_report_writer = None
        self.csv_header_written = False
    
    def generate_pipeline_combinations(self):
        """
        Generate all possible pipeline combinations based on parameters.
        
        Returns:
            list: List of tuples containing (estimator, scaler, feature_selector, searcher)
        """
        all_pipelines = list(itertools.product(*[
            filter(lambda x: False if x in self.ignore_estimator else True, ESTIMATOR_NAMES),
            filter(lambda x: False if x in self.ignore_scaler else True, SCALER_NAMES),
            filter(lambda x: False if x in self.ignore_feature_selector else True, FEATURE_SELECTOR_NAMES),
            filter(lambda x: False if x in self.ignore_searcher else True, SEARCHER_NAMES),
        ]))
        
        if not len(all_pipelines):
            raise ValueError('No pipelines to run with the current configuration')
            
        return all_pipelines
    
    def create_pipeline(self, estimator, scaler, feature_selector, searcher, y_train):
        """
        Create a pipeline for the given configuration.
        
        Args:
            estimator (str): Estimator key
            scaler (str): Scaler key  
            feature_selector (str): Feature selector key
            searcher (str): Searcher key
            y_train (array): Training labels
            
        Returns:
            tuple: (pipeline, total_fits)
        """
        steps = []

        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        if feature_selector and FEATURE_SELECTORS[feature_selector]:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        steps.append(('debug', Debug()))

        scoring = self.scorers if self.scorers else ['accuracy']

        # Check if this is a multiclass problem
        n_classes = len(np.unique(y_train))
        
        scorers = {}
        for scorer in scoring:
            if scorer == 'roc_auc' and n_classes > 2:
                # Use roc_auc_ovr for multiclass problems
                scorers[scorer] = 'roc_auc_ovr'
            else:
                scorers[scorer] = scorer

        search_step = SEARCHERS[searcher](estimator, scorers, self.shuffle, self.custom_hyper_parameters, y_train)

        steps.append(('estimator', search_step[0]))

        return (Pipeline(steps), search_step[1])
    
    def train_model(self, pipeline, feature_names, x_train, y_train):
        """
        Train a model using the provided pipeline.
        
        Args:
            pipeline: Sklearn pipeline
            feature_names (list): List of feature names
            x_train (array): Training features
            y_train (array): Training labels
            
        Returns:
            dict: Model training results
        """
        MAX_FEATURES_SHOWN = 5
        
        start = timer()
        features = {}
        feature_scores = None
        selected_features = feature_names

        pipeline.fit(x_train, y_train)

        if 'feature_selector' in pipeline.named_steps:
            feature_selector_type = pipeline.named_steps['feature_selector'].__class__.__module__

            if 'univariate_selection' in feature_selector_type:
                feature_scores = pipeline.named_steps['feature_selector'].scores_
                feature_scores = pd.DataFrame({'scores': feature_scores, 'selected': pipeline.named_steps['feature_selector'].get_support()}, index=feature_names)
                feature_scores = feature_scores[feature_scores['selected'] == True].drop(columns=['selected'])
                features = pd.Series(pipeline.named_steps['feature_selector'].get_support(),
                                     index=feature_names)
                selected_features = features[features == True].axes[0]

            elif 'processors.rffi' in feature_selector_type:
                most_important = pipeline.named_steps['feature_selector'].get_top_features()
                most_important_names =\
                    [feature_names[most_important[i]] for i in range(len(most_important))]
                feature_scores = pipeline.named_steps['feature_selector'].model.feature_importances_
                feature_scores = pd.DataFrame({'scores': feature_scores, 'selected': list(i in most_important_names for i in feature_names)}, index=feature_names)
                feature_scores = feature_scores[feature_scores['selected'] == True].drop(columns=['selected'])
                features = pd.Series((i in most_important_names for i in feature_names),
                                     index=feature_names)
                selected_features = features[features == True].axes[0]

        if feature_scores is not None:
            total_score = feature_scores['scores'].sum()
            feature_scores['scores'] = round(feature_scores['scores'] / total_score, 4)
            feature_scores = json.dumps(dict(feature_scores['scores'].sort_values(ascending=False)))
        else:
            feature_scores = ""

        print('\tFeatures used: ' + ', '.join(selected_features[:MAX_FEATURES_SHOWN]) +
              ('...' if len(selected_features) > MAX_FEATURES_SHOWN else ''))

        train_time = timer() - start
        print('\tTraining time is {:.4f} seconds'.format(train_time), '\n')

        return {
            'features': features,
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'train_time': train_time
        }
    
    def refit_candidates(self, pipeline, features, estimator, scorer, x_train, y_train):
        """
        Generate refitted model candidates for different scorers.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator (str): Estimator key
            scorer (str): Scorer key
            x_train (array): Training features
            y_train (array): Training labels
            
        Returns:
            list: List of candidate models
        """
        MODELS_TO_EVALUATE = 2
        
        # Transform values based on the pipeline
        x_train = preprocess(features, pipeline, x_train)

        results = pipeline.named_steps['estimator'].cv_results_

        # Select the top search results
        sorted_results = sorted(
            range(len(results['rank_test_%s' % scorer])),
            key=lambda i: results['rank_test_%s' % scorer][i]
        )[:MODELS_TO_EVALUATE]

        models = []

        for position, index in enumerate(sorted_results):
            best_params_ = results['params'][index]

            print('\t#%d %s: %.7g (sd=%.7g)'
                  % (position+1, SCORER_NAMES[scorer], results['mean_test_%s' % scorer][index],
                     results['std_test_%s' % scorer][index]))
            print('\t#%d %s parameters:' % (position+1, SCORER_NAMES[scorer]),
                  json.dumps(best_params_, indent=4, sort_keys=True).replace('\n', '\n\t'))

            # Handle XGBoost configuration properly
            if estimator == 'gb':
                n_classes = len(pd.Series(y_train).unique())
                base_estimator = get_xgb_classifier(n_classes)
            else:
                base_estimator = ESTIMATORS[estimator]

            model = clone(base_estimator).set_params(
                **best_params_).fit(x_train, y_train)

            models.append({
                'best_estimator': model,
                'best_params': best_params_
            })

        return models
    
    def initialize_reports(self):
        """Initialize CSV report files for results."""
        self.report = open(self.output_path + '/report.csv', 'w+')
        self.report_writer = csv.writer(self.report)
        
        self.performance_report = open(self.output_path + '/performance_report.csv', 'w+')
        self.performance_report_writer = csv.writer(self.performance_report)
        self.performance_report_writer.writerow(['key', 'train_time (s)'])
    
    def write_result_to_csv(self, result):
        """
        Write a result dictionary to the CSV report.
        
        Args:
            result (dict): Result dictionary to write
        """
        if not self.csv_header_written:
            self.report_writer.writerow(result.keys())
            self.csv_header_written = True
        
        self.report_writer.writerow(list([str(i) for i in result.values()]))
    
    def save_model_archives(self):
        """Save all models in compressed archives."""
        models_dir = f"{self.output_path}/models"
        os.makedirs(models_dir, exist_ok=True)
        
        print(f'Saving {len(self.main_models)} main models to compressed archive...')
        
        # Save main models archive
        if self.main_models:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save individual models to temp directory
                for key, model in self.main_models.items():
                    temp_path = f"{temp_dir}/{key}.joblib"
                    dump(model, temp_path)
                
                # Create compressed archive
                with tarfile.open(f"{models_dir}/main_models.tar.gz", "w:gz") as tar:
                    tar.add(temp_dir, arcname="main_models")
            
            print(f'Main models archive saved: {models_dir}/main_models.tar.gz')
        
        # Save OvR models archive (if any)
        if self.ovr_models:
            print(f'Saving {len(self.ovr_models)} OvR models to compressed archive...')
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for key, model in self.ovr_models.items():
                    temp_path = f"{temp_dir}/{key}.joblib"
                    dump(model, temp_path)
                
                with tarfile.open(f"{models_dir}/ovr_models.tar.gz", "w:gz") as tar:
                    tar.add(temp_dir, arcname="ovr_models")
            
            print(f'OvR models archive saved: {models_dir}/ovr_models.tar.gz')
    
    def load_models_from_archives(self):
        """Load models from compressed archives back into classifier instance."""
        models_dir = f"{self.output_path}/models"
        
        # Load main models
        main_archive_path = f"{models_dir}/main_models.tar.gz"
        if os.path.exists(main_archive_path):
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(main_archive_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                
                main_models_dir = f"{temp_dir}/main_models"
                if os.path.exists(main_models_dir):
                    for filename in os.listdir(main_models_dir):
                        if filename.endswith('.joblib'):
                            model_key = filename[:-7]  # Remove .joblib extension
                            model_path = f"{main_models_dir}/{filename}"
                            self.main_models[model_key] = load(model_path)
            
            print(f'Loaded {len(self.main_models)} main models from archive')
        
        # Load OvR models
        ovr_archive_path = f"{models_dir}/ovr_models.tar.gz"
        if os.path.exists(ovr_archive_path):
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(ovr_archive_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                
                ovr_models_dir = f"{temp_dir}/ovr_models"
                if os.path.exists(ovr_models_dir):
                    for filename in os.listdir(ovr_models_dir):
                        if filename.endswith('.joblib'):
                            model_key = filename[:-7]  # Remove .joblib extension
                            model_path = f"{ovr_models_dir}/{filename}"
                            self.ovr_models[model_key] = load(model_key)
            
            print(f'Loaded {len(self.ovr_models)} OvR models from archive')
    
    
    def predict_ensemble(self, data, model_keys, vote_type='majority'):
        """
        Make predictions using ensemble of models.
        
        Args:
            data: Input data for prediction
            model_keys (list): List of model keys to use in ensemble
            vote_type (str): Type of voting ('majority', 'weighted', etc.)
            
        Returns:
            dict: Ensemble prediction results
        """
        predictions_list = []
        probabilities_list = []
        
        for model_key in model_keys:
            if model_key in self.main_models:
                model = self.main_models[model_key]
                pred_result = self.predict(data, model_key)
                predictions_list.append(pred_result['predicted'])
                probabilities_list.append(pred_result['probability'])
        
        if not predictions_list:
            raise ValueError("No valid models found for ensemble")
        
        # Simple majority voting
        predictions_array = np.array(predictions_list)
        if vote_type == 'majority':
            ensemble_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions_array
            )
        else:
            # Default to majority for now
            ensemble_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions_array
            )
        
        # Average probabilities
        ensemble_probabilities = np.mean(probabilities_list, axis=0)
        
        return {
            'predicted': ensemble_predictions.tolist(),
            'probability': ensemble_probabilities.tolist()
        }
    
    def get_generalization_metrics(self, data, model_key, label_column, threshold=0.5):
        """
        Get generalization metrics for a specific model.
        
        Args:
            data: Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            threshold (float): Prediction threshold
            
        Returns:
            dict: Generalization metrics
        """
        if model_key not in self.main_models:
            raise KeyError(f"Model {model_key} not found")
        
        model = self.main_models[model_key]
        
        # Extract features and labels from data
        if isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'], columns=data.get('columns', []))
            X = df.drop(columns=[label_column]).values
            y = df[label_column].values
        else:
            raise ValueError("Data format not supported")
        
        # Use the model's evaluation methods
        return self.evaluate_generalization(None, None, model, X, y, None)
    
    def get_additional_roc(self, data, model_key, label_column):
        """
        Get ROC metrics for a specific model.
        
        Args:
            data: Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            
        Returns:
            dict: ROC metrics
        """
        if model_key not in self.main_models:
            raise KeyError(f"Model {model_key} not found")
        
        model = self.main_models[model_key]
        
        # Extract features and labels from data
        if isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'], columns=data.get('columns', []))
            X = df.drop(columns=[label_column]).values
            y = df[label_column].values
        else:
            raise ValueError("Data format not supported")
        
        # Use the model's ROC evaluation method
        return self.evaluate_roc(None, None, model, X, y)
    
    def get_additional_precision(self, data, model_key, label_column):
        """
        Get precision-recall metrics for a specific model.
        
        Args:
            data: Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            
        Returns:
            dict: Precision-recall metrics
        """
        if model_key not in self.main_models:
            raise KeyError(f"Model {model_key} not found")
        
        model = self.main_models[model_key]
        
        # Extract features and labels from data
        if isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'], columns=data.get('columns', []))
            X = df.drop(columns=[label_column]).values
            y = df[label_column].values
        else:
            raise ValueError("Data format not supported")
        
        # Use the model's precision-recall evaluation method
        return self.evaluate_precision_recall(None, None, model, X, y)
    
    def get_additional_reliability(self, data, model_key, label_column):
        """
        Get reliability metrics for a specific model.
        
        Args:
            data: Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            
        Returns:
            dict: Reliability metrics
        """
        if model_key not in self.main_models:
            raise KeyError(f"Model {model_key} not found")
        
        model = self.main_models[model_key]
        
        # Extract features and labels from data
        if isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'], columns=data.get('columns', []))
            X = df.drop(columns=[label_column]).values
            y = df[label_column].values
        else:
            raise ValueError("Data format not supported")
        
        # Use the model's reliability evaluation method
        return self.evaluate_reliability(None, None, model, X, y)
    
    def list_available_pipelines(self):
        """
        List available pipeline configurations based on current parameters.
        
        Returns:
            list: Available pipeline configurations
        """        
        # Filter based on ignored parameters
        available_estimators = [x for x in ESTIMATOR_NAMES if x not in self.ignore_estimator]
        available_scalers = [x for x in SCALER_NAMES if x not in self.ignore_scaler]
        available_feature_selectors = [x for x in FEATURE_SELECTOR_NAMES if x not in self.ignore_feature_selector]
        available_searchers = [x for x in SEARCHER_NAMES if x not in self.ignore_searcher]
        
        return {
            'estimators': available_estimators,
            'scalers': available_scalers,
            'feature_selectors': available_feature_selectors,
            'searchers': available_searchers,
            'scorers': self.scorers
        }
    
    def create_static_model(self, model_key, parameters, features, dataset_folder, label_column, threshold=0.5):
        """
        Create a static copy of a selected model.
        
        Args:
            model_key (str): Key identifying the model
            parameters (dict): Model parameters
            features (list): Selected features
            dataset_folder (str): Path to dataset folder
            label_column (str): Target column name
            threshold (float): Prediction threshold
            
        Returns:
            dict: Generalization results for the static model
        """
        if model_key not in self.main_models:
            raise KeyError(f"Model {model_key} not found")
        
        # Save the selected model as a static copy
        model = self.main_models[model_key]
        
        # Save model to pipeline.joblib for compatibility with existing code
        dump(model, f"{self.output_path}/pipeline.joblib")
        
        # Save features information
        with open(f"{self.output_path}/features.json", 'w') as f:
            json.dump(features, f)
        
        # Save parameters
        with open(f"{self.output_path}/parameters.json", 'w') as f:
            json.dump(parameters, f)
        
        return {
            'model_key': model_key,
            'features': features,
            'parameters': parameters,
            'output_path': self.output_path
        }
    
    def list_available_models(self):
        """
        List all available trained models.
        
        Returns:
            dict: Information about available models
        """
        return {
            'main_models': list(self.main_models.keys()),
            'ovr_models': list(self.ovr_models.keys()),
            'total_main': len(self.main_models),
            'total_ovr': len(self.ovr_models)
        }
    
    def save_class_results(self, class_data, model_key):
        """
        Save class-specific results to compressed pickle file.
        
        Args:
            class_data (dict): Class-specific data to save
            model_key (str): Model key for filename
        """
        
        class_results_dir = self.output_path + '/class_results'
        os.makedirs(class_results_dir, exist_ok=True)
        
        # Save as compressed pickle file
        filename = f"{class_results_dir}/{model_key}_class_results.pkl.gz"
        with gzip.open(filename, 'wb') as f:
            pickle.dump(class_data, f)
        
        print(f"Saved class results to {filename}")
    
    def finalize_reports(self, start_time, n_classes):
        """
        Finalize and close report files, save metadata.
        
        Args:
            start_time (float): Start time of the training process
            n_classes (int): Number of classes in the dataset
        """
        train_time = timer() - start_time
        print('\tTotal run time is {:.4f} seconds'.format(train_time), '\n')
        self.performance_report_writer.writerow(['total', train_time])
        
        self.report.close()
        self.performance_report.close()
        
        print('Total fits generated', sum(self.total_fits.values()))
        
        # Save model archives
        self.save_model_archives()
        
        # Update metadata
        metadata = {
            'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'fits': self.total_fits,
            'ovr_models_count': len(self.ovr_models),
            'main_models_count': len(self.main_models),
            'n_classes': n_classes
        }
        
        if self.output_path != '.':
            metadata_path = self.output_path + '/metadata.json'
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
    
    def create_base_result(self, key, estimator, scaler, feature_selector, searcher, scorer, n_classes, position=0):
        """
        Create a base result dictionary with common fields.
        
        Args:
            key (str): Model key
            estimator (str): Estimator key
            scaler (str): Scaler key
            feature_selector (str): Feature selector key
            searcher (str): Searcher key
            scorer (str): Scorer key
            n_classes (int): Number of classes
            position (int): Position in candidate list
            
        Returns:
            dict: Base result dictionary
        """
        return {
            'key': key + '__' + str(position),
            'class_type': 'multiclass' if n_classes > 2 else 'binary',
            'class_index': None,
            'scaler': SCALER_NAMES[scaler],
            'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
            'algorithm': ESTIMATOR_NAMES[estimator],
            'searcher': SEARCHER_NAMES[searcher],
            'scorer': SCORER_NAMES[scorer],
        }
    
    def evaluate_model_complete(self, pipeline, features, estimator, x_val, y_val, x_test, y_test, labels):
        """
        Complete model evaluation using all evaluation methods.
        
        This method calls all the abstract evaluation methods and combines their results.
        Subclasses can override this if they need custom evaluation logic.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_val (array): Validation features (for hyperparameter evaluation)
            y_val (array): Validation labels (for hyperparameter evaluation)
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            labels (list): Class labels
            
        Returns:
            dict: Complete evaluation results
        """
        # Get generalization results using test data
        result = self.evaluate_generalization(pipeline, features, estimator, x_test, y_test, labels)
        
        # Add ROC curve data for validation set (used for training ROC AUC)
        val_roc = self.evaluate_roc(pipeline, features, estimator, x_val, y_val)
        result.update({
            'test_fpr': val_roc.get('fpr'),
            'test_tpr': val_roc.get('tpr'),
            'training_roc_auc': val_roc.get('roc_auc')
        })
        
        # Calculate ROC delta (difference between generalization and training)
        if 'roc_auc' in result and 'training_roc_auc' in result:
            if result['roc_auc'] is not None and result['training_roc_auc'] is not None:
                result['roc_delta'] = round(abs(result['roc_auc'] - result['training_roc_auc']), 4)
        
        # Add generalization ROC curve data using test data
        test_roc = self.evaluate_roc(pipeline, features, estimator, x_test, y_test)
        result.update({
            'generalization_fpr': test_roc.get('fpr'),
            'generalization_tpr': test_roc.get('tpr')
        })
        
        # Add reliability metrics using test data
        result.update(self.evaluate_reliability(pipeline, features, estimator, x_test, y_test))
        
        # Add precision-recall metrics using test data
        result.update(self.evaluate_precision_recall(pipeline, features, estimator, x_test, y_test))
        
        return result
    
    def save_class_results(self, class_data, model_key):
        """Save class-specific results with compression"""        
        # Ensure output directory exists
        class_results_dir = self.output_path + '/class_results'
        os.makedirs(class_results_dir, exist_ok=True)
        
        filepath = f"{class_results_dir}/{model_key}.pkl.gz"
        try:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(class_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving class results for {model_key}: {e}")

    @staticmethod
    def load_class_results(class_results_dir, model_key):
        """Load class-specific results"""        
        filepath = f"{class_results_dir}/{model_key}.pkl.gz"
        if os.path.exists(filepath):
            try:
                with gzip.open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading class results for {model_key}: {e}")
                return None
        return None

    @staticmethod
    def get_available_models_with_class_results(class_results_dir):
        """Get list of models that have class-specific results"""

        
        if not os.path.exists(class_results_dir):
            return []
        
        models = []
        try:
            for filename in os.listdir(class_results_dir):
                if filename.endswith('.pkl.gz'):
                    models.append(filename.replace('.pkl.gz', ''))
        except Exception as e:
            print(f"Error listing class results: {e}")
            return []
        
        return models

    @staticmethod
    def cleanup_class_results(class_results_dir):
        """Remove all class results for a job"""

        
        if os.path.exists(class_results_dir):
            try:
                shutil.rmtree(class_results_dir)
            except Exception as e:
                print(f"Error cleaning up class results: {e}")
