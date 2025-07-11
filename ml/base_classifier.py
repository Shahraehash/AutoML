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
    
from .processors.estimators import ESTIMATOR_NAMES, ESTIMATORS, get_xgb_classifier
from .processors.feature_selection import FEATURE_SELECTOR_NAMES, FEATURE_SELECTORS
from .processors.scalers import SCALER_NAMES, SCALERS
from .processors.searchers import SEARCHER_NAMES, SEARCHERS
from .processors.scorers import SCORER_NAMES
from .processors.debug import Debug
from .utils.preprocess import preprocess
from .utils.utils import model_key_to_name
from .utils.stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci


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
        self.MODELS_TO_EVALUATE = 2
        self.MAX_FEATURES_SHOWN = 5 
        self.STANDARD_CSV_FIELDS = {
            # Basic model information
                'key': None,
                'class_type': None,
                'class_index': None,
                'scaler': None,
                'feature_selector': None,
                'algorithm': None,
                'searcher': None,
                'scorer': None,
                
            # Generalization metrics
                'accuracy': None,
                'acc_95_ci': None,
                'mcc': None,
                'avg_sn_sp': None,
                'roc_auc': None,
                'roc_auc_95_ci': None,
                'f1': None,
                'sensitivity': None,
                'sn_95_ci': None,
                'specificity': None,
                'sp_95_ci': None,
                'prevalence': None,
                'pr_95_ci': None,
                'ppv': None,
                'ppv_95_ci': None,
                'npv': None,
                'npv_95_ci': None,
                'tn': None,
                'tp': None,
                'fn': None,
                'fp': None,
                
                # ROC data
                'test_fpr': None,
                'test_tpr': None,
                'training_roc_auc': None,
                'roc_delta': None,
                'generalization_fpr': None,
                'generalization_tpr': None,
                
                # Reliability data
                'brier_score': None,
                'fop': None,
                'mpv': None,
                
                # Precision-recall data
                'precision': None,
                'recall': None,
                
                # Model-specific data
                'selected_features': None,
                'feature_scores': None,
                'best_params': None
        }
    


    def generate_model(self, pipeline, feature_names, x_train, y_train):
        """Define the generic method to generate the best model for the provided estimator"""

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

    def refit_model(self, pipeline, features, estimator, scoring, x_train, y_train):
        """
        Determine the best model based on the provided scoring method
        and the fitting pipeline results.
        """

        # Transform values based on the pipeline
        x_train = preprocess(features, pipeline, x_train)

        results = pipeline.named_steps['estimator'].cv_results_

        # Select the top search results
        sorted_results = sorted(
            range(len(results['rank_test_%s' % scoring])),
            key=lambda i: results['rank_test_%s' % scoring][i]
        )[:self.MODELS_TO_EVALUATE]

        models = []

        for position, index in enumerate(sorted_results):
            best_params_ = results['params'][index]

            print('\t#%d %s: %.7g (sd=%.7g)'
                % (position+1, SCORER_NAMES[scoring], results['mean_test_%s' % scoring][index],
                    results['std_test_%s' % scoring][index]))
            print('\t#%d %s parameters:' % (position+1, SCORER_NAMES[scoring]),
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

    def get_available_models_with_class_results(self, class_results_dir):
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

    def save_class_results(self, class_data, output_dir, model_key):
        """Save class-specific results with compression"""
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = f"{output_dir}/{model_key}.pkl.gz"
        try:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(class_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving class results for {model_key}: {e}")

    def load_class_results(self, class_results_dir, model_key):
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

    def save_model_archives(self, main_models, ovr_models, output_path):
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

