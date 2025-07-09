"""
Unified Classifier Manager

This module provides a high-level orchestration layer that manages classifier instances,
coordinates training operations, and provides a clean interface for all ML operations.
It acts as the bridge between the frontend API and the underlying classifier classes
and file operations.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
import itertools

from .processors.estimators import ESTIMATOR_NAMES
from .processors.feature_selection import FEATURE_SELECTOR_NAMES
from .processors.scalers import SCALER_NAMES
from .processors.searchers import SEARCHER_NAMES
from .processors.scorers import SCORER_NAMES
from .classifiers.binary_classifier import BinaryClassifier
from .classifiers.multiclass_macro_classifier import MulticlassMacroClassifier
from .classifiers.multiclass_ovr_classifier import OvRClassifier
from .classifiers.static_model_classifier import StaticModelClassifier
from .classifiers.tandem_model_classifier import TandemModelClassifier
from .classifiers.ensemble_model_classifier import EnsembleModelClassifier
from .job_result_accessor import JobResultAccessor, ResultNotFoundError, ModelNotFoundError
from .import_data import import_data, import_csv


class ClassifierNotTrainedException(Exception):
    """Exception raised when attempting operations on untrained classifier."""
    pass


class UnifiedClassifierManager:
    """
    High-level manager for classifier operations.
    
    This class provides a unified interface for all classifier operations including
    training, prediction, evaluation, and result access. It coordinates between
    the classifier classes and the JobResultAccessor for clean separation of concerns.
    """
    
    def __init__(self, job_folder: str, parameters: Dict[str, Any], update_function=lambda x, y: None):
        """
        Initialize the Unified Classifier Manager.
        
        Args:
            job_folder (str): Path to the job folder for storing results
            parameters (dict): Configuration parameters for the classifier
            update_function (callable): Callback function for progress updates
        """
        self.job_folder = job_folder
        self.parameters = parameters
        self.update_function = update_function
        
        # Initialize result accessor for file operations
        self.result_accessor = JobResultAccessor(job_folder)
        
        # Classifier instance will be created when needed
        self.classifier_instance = None
        self.classifier_type = None
        self.is_trained = False
        self.n_classes = None
        
    def create_and_train_models(self, train_set: str, test_set: str, label_column: str, labels: Optional[List[str]] = None) -> bool:
        """
        Create appropriate classifier and train models.
        
        Args:
            train_set (str): Path to training data file
            test_set (str): Path to test data file
            label_column (str): Name of the target column
            labels (list, optional): Class labels
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Import and prepare data
            (x_train, x_val, y_train, y_val, x_test, y_test, feature_names, metadata) = \
                import_data(train_set, test_set, label_column)
            
            # Determine number of classes to choose appropriate classifier
            self.n_classes = len(np.unique(y_train))
            print(f'Detected {self.n_classes} classes in target column')
            
            # Choose classifier based on number of classes and parameters
            reoptimize_ovr = self.parameters.get('reoptimize_ovr', 'false').lower() == 'true'
            
            if self.n_classes == 2:
                print('Using BinaryClassifier for 2-class problem')
                self.classifier_instance = BinaryClassifier(self.parameters, self.job_folder, self.update_function)
                self.classifier_type = 'binary'
            elif self.n_classes > 2 and reoptimize_ovr:
                print('Using OvRClassifier for multiclass problem with OvR re-optimization')
                self.classifier_instance = OvRClassifier(self.parameters, self.job_folder, self.update_function)
                self.classifier_type = 'ovr'
            elif self.n_classes > 2:
                print('Using MulticlassMacroClassifier for multiclass problem')
                self.classifier_instance = MulticlassMacroClassifier(self.parameters, self.job_folder, self.update_function)
                self.classifier_type = 'multiclass_macro'
            else:
                print(f'Invalid number of classes: {self.n_classes}')
                return False
            
            # Train the classifier
            success = self.classifier_instance.fit(
                x_train, x_val, y_train, y_val, x_test, y_test, feature_names, labels
            )
            
            self.is_trained = success
            
            if success:
                print(f'Training completed successfully with {self.classifier_type} classifier')
            else:
                print('Training failed')
                
            return success
            
        except Exception as e:
            print(f'Error during training: {str(e)}')
            return False
    
    def get_main_results(self) -> Dict[str, Any]:
        """
        Get main training results.
        
        Returns:
            dict: Main results including metadata
        """
        return self.result_accessor.get_main_results()
    
    def get_class_results(self, class_index: int, model_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get class-specific results for multiclass models.
        
        Args:
            class_index (int): Index of the class
            model_key (str, optional): Specific model key
            
        Returns:
            dict: Class-specific results
        """
        return self.result_accessor.get_class_results(class_index, model_key)
    
    def list_available_pipelines(self) -> Dict[str, Any]:
        """
        List available pipeline configurations.
        
        Returns:
            dict: Available pipeline configurations
        """
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.list_available_pipelines()
    
    @staticmethod
    def list_pipeline_configurations(parameters: Dict[str, Any]) -> List:
        """
        List available pipeline configurations based on parameters.
        This is a static method that doesn't require a trained classifier.
        
        Args:
            parameters (dict): Configuration parameters
            
        Returns:
            list: Available pipeline configurations
        """
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
    
    def list_available_models(self, class_index: Optional[int] = None) -> Dict[str, Any]:
        """
        List available trained models.
        
        Args:
            class_index (int, optional): Class index for OvR models
            
        Returns:
            dict: Information about available models
        """
        if self.is_trained:
            # Use classifier instance method
            return self.classifier_instance.list_available_models()
        else:
            # Fallback to result accessor for loading from files
            return self.result_accessor.list_available_models(class_index)
    
    def get_available_class_models(self) -> Dict[str, Any]:
        """
        Get list of models that have class-specific results.
        
        Returns:
            dict: Information about models with class-specific results
        """
        return self.result_accessor.get_available_class_models()
    
    def make_predictions(self, data: Union[np.ndarray, List], model_key: str, threshold: float = 0.5, class_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            data: Input data for prediction
            model_key (str): Key identifying the model
            threshold (float): Prediction threshold
            class_index (int, optional): Class index for OvR models
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.predict(data, model_key, threshold, class_index)
    
    def make_ensemble_predictions(self, data: Union[np.ndarray, List], model_keys: List[str], vote_type: str = 'majority') -> Dict[str, Any]:
        """
        Make predictions using ensemble of models.
        
        Args:
            data: Input data for prediction
            model_keys (list): List of model keys to use in ensemble
            vote_type (str): Type of voting ('majority', 'weighted', etc.)
            
        Returns:
            dict: Ensemble prediction results
        """
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.predict_ensemble(data, model_keys, vote_type)
    
    def make_ensemble_generalization(self, total_models: int, dataset_folder: str, label_column: str) -> Dict[str, Any]:
        """
        Generate generalization results for ensemble models.
        
        Args:
            total_models (int): Number of models in the ensemble
            dataset_folder (str): Path to dataset folder containing test data
            label_column (str): Name of the target column
            
        Returns:
            dict: Ensemble generalization results with soft and hard voting
        """
        # Load test data
        x_test, y_test, feature_names, _, _ = import_csv(dataset_folder + '/test.csv', label_column)
        data = pd.DataFrame(x_test, columns=feature_names)

        # Get ensemble predictions for both voting types
        model_keys = [f'ensemble{i}' for i in range(total_models)]
        
        soft_result = self.make_ensemble_predictions(data, model_keys, 'soft')
        hard_result = self.make_ensemble_predictions(data, model_keys, 'hard')

        # Determine if this is binary or multi-class
        unique_labels = sorted(y_test.unique())
        if len(unique_labels) == 2:
            labels = ['No ' + label_column, label_column]
        else:
            labels = [f'Class {int(cls)}' for cls in unique_labels]

        # Use the classifier's generalization_report method
        return {
            'soft_generalization': self.classifier_instance.generalization_report(labels, y_test, soft_result['predicted'], soft_result['probability']),
            'hard_generalization': self.classifier_instance.generalization_report(labels, y_test, hard_result['predicted'], hard_result['probability'])
        }
    
    def get_generalization_metrics(self, data: Dict[str, Any], model_key: str, label_column: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Get generalization metrics for a model.
        
        Args:
            data (dict): Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            threshold (float): Prediction threshold
            
        Returns:
            dict: Generalization metrics
        """
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.get_generalization_metrics(data, model_key, label_column, threshold)
    
    def get_additional_roc_metrics(self, data: Dict[str, Any], model_key: str, label_column: str) -> Dict[str, Any]:
        """
        Get ROC metrics for a specific model.
        
        Args:
            data (dict): Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            
        Returns:
            dict: ROC metrics
        """
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.get_additional_roc(data, model_key, label_column)
    
    def get_additional_precision_metrics(self, data: Dict[str, Any], model_key: str, label_column: str) -> Dict[str, Any]:
        """
        Get precision-recall metrics for a specific model.
        
        Args:
            data (dict): Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            
        Returns:
            dict: Precision-recall metrics
        """
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.get_additional_precision(data, model_key, label_column)
    
    def get_additional_reliability_metrics(self, data: Dict[str, Any], model_key: str, label_column: str) -> Dict[str, Any]:
        """
        Get reliability metrics for a specific model.
        
        Args:
            data (dict): Test data
            model_key (str): Key identifying the model
            label_column (str): Target column name
            
        Returns:
            dict: Reliability metrics
        """
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.get_additional_reliability(data, model_key, label_column)
    
    def create_static_model(self, model_key: str, parameters: Dict[str, Any], features: List[str], 
                           dataset_folder: str, label_column: str, threshold: float = 0.5) -> Dict[str, Any]:
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
        if not self.is_trained:
            raise ClassifierNotTrainedException("Classifier must be trained first")
        
        return self.classifier_instance.create_static_model(
            model_key, parameters, features, dataset_folder, label_column, threshold
        )
    
    def export_model(self, model_key: str, class_index: Optional[int] = None, threshold: float = 0.5):
        """
        Export model for deployment.
        
        Args:
            model_key (str): Key identifying the model
            class_index (int, optional): Class index for OvR models
            threshold (float): Prediction threshold
            
        Returns:
            BytesIO: ZIP file containing model and dependencies
        """
        return self.result_accessor.create_model_export_zip(model_key, class_index, threshold)
    
    def export_results_csv(self, class_index: Optional[int] = None):
        """
        Export results as CSV.
        
        Args:
            class_index (int, optional): Filter by class index
            
        Returns:
            tuple: (DataFrame, filename)
        """
        return self.result_accessor.get_csv_export_data(class_index)
    
    def get_starred_models(self) -> List[str]:
        """
        Get list of starred models.
        
        Returns:
            list: Starred model keys
        """
        return self.result_accessor.get_starred_models()
    
    def update_starred_models(self, models_to_add: Optional[List[str]] = None, 
                            models_to_remove: Optional[List[str]] = None):
        """
        Update starred models list.
        
        Args:
            models_to_add (list, optional): Models to add to starred list
            models_to_remove (list, optional): Models to remove from starred list
        """
        self.result_accessor.update_starred_models(models_to_add, models_to_remove)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get job metadata.
        
        Returns:
            dict: Job metadata
        """
        return self.result_accessor.metadata
    
    def cleanup(self):
        """
        Clean up resources and temporary files.
        """
        if self.classifier_instance:
            # Any classifier-specific cleanup if needed
            pass
        
        # Reset state
        self.classifier_instance = None
        self.is_trained = False
        self.n_classes = None
        self.classifier_type = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the classifier manager.
        
        Returns:
            dict: Status information
        """
        return {
            'is_trained': self.is_trained,
            'classifier_type': self.classifier_type,
            'n_classes': self.n_classes,
            'job_folder': self.job_folder,
            'has_results': self.result_accessor.file_exists('report.csv'),
            'has_models': (
                self.result_accessor.file_exists('models/main_models.tar.gz') or
                self.result_accessor.file_exists('models/ovr_models.tar.gz')
            )
        }
    
    def load_static_model(self, model_path: str) -> bool:
        """
        Load a static model (pipeline.joblib) for prediction/evaluation.
        
        Args:
            model_path (str): Path to the static model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create StaticModelClassifier instance
            self.classifier_instance = StaticModelClassifier(self.parameters, self.job_folder, self.update_function)
            
            # Load the static model
            success = self.classifier_instance.load_model(model_path)
            
            if success:
                self.classifier_type = 'static'
                self.is_trained = True
                self.n_classes = self.classifier_instance.n_classes
                print(f"Successfully loaded static model from {model_path}")
            else:
                print(f"Failed to load static model from {model_path}")
                
            return success
            
        except Exception as e:
            print(f"Error loading static model: {e}")
            return False
    
    def get_static_generalization_metrics(self, data: Dict[str, Any], model_key: str, label_column: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Get generalization metrics for a static model.
        
        Args:
            data (dict): Test data
            model_key (str): Model identifier
            label_column (str): Target column name
            threshold (float): Prediction threshold
            
        Returns:
            dict: Generalization metrics
        """
        if not self.is_trained or self.classifier_type != 'static':
            raise ClassifierNotTrainedException("Static model must be loaded first")
        
        return self.classifier_instance.get_generalization_metrics(data, label_column, threshold)
    
    def get_static_reliability_metrics(self, data: Dict[str, Any], model_key: str, label_column: str) -> Dict[str, Any]:
        """
        Get reliability metrics for a static model.
        
        Args:
            data (dict): Test data
            model_key (str): Model identifier
            label_column (str): Target column name
            
        Returns:
            dict: Reliability metrics
        """
        if not self.is_trained or self.classifier_type != 'static':
            raise ClassifierNotTrainedException("Static model must be loaded first")
        
        return self.classifier_instance.get_reliability_metrics(data, label_column)
    
    def get_static_precision_metrics(self, data: Dict[str, Any], model_key: str, label_column: str) -> Dict[str, Any]:
        """
        Get precision-recall metrics for a static model.
        
        Args:
            data (dict): Test data
            model_key (str): Model identifier
            label_column (str): Target column name
            
        Returns:
            dict: Precision-recall metrics
        """
        if not self.is_trained or self.classifier_type != 'static':
            raise ClassifierNotTrainedException("Static model must be loaded first")
        
        return self.classifier_instance.get_precision_metrics(data, label_column)
    
    def get_static_roc_metrics(self, data: Dict[str, Any], model_key: str, label_column: str) -> Dict[str, Any]:
        """
        Get ROC metrics for a static model.
        
        Args:
            data (dict): Test data
            model_key (str): Model identifier
            label_column (str): Target column name
            
        Returns:
            dict: ROC metrics
        """
        if not self.is_trained or self.classifier_type != 'static':
            raise ClassifierNotTrainedException("Static model must be loaded first")
        
        return self.classifier_instance.get_roc_metrics(data, label_column)
    
    def load_tandem_models(self, folder: str) -> bool:
        """
        Load tandem models (NPV + PPV) for prediction.
        
        Args:
            folder (str): Path to folder containing tandem models
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create TandemModelClassifier instance
            self.classifier_instance = TandemModelClassifier(self.job_folder, self.parameters)
            
            # Load the tandem models
            success = self.classifier_instance.load_tandem_models(folder)
            
            if success:
                self.classifier_type = 'tandem'
                self.is_trained = True
                self.n_classes = 2  # Tandem models are binary
                print(f"Successfully loaded tandem models from {folder}")
            else:
                print(f"Failed to load tandem models from {folder}")
                
            return success
            
        except Exception as e:
            print(f"Error loading tandem models: {e}")
            return False
    
    def make_tandem_predictions(self, data: Union[np.ndarray, List], features: List[str]) -> Dict[str, Any]:
        """
        Make predictions using tandem models.
        
        Args:
            data: Input data for prediction
            features: List of feature names
            
        Returns:
            dict: Tandem prediction results
        """
        if not self.is_trained or self.classifier_type != 'tandem':
            raise ClassifierNotTrainedException("Tandem models must be loaded first")
        
        return self.classifier_instance.predict_tandem(data, features)
    
    def load_ensemble_models(self, folder: str, total_models: int) -> bool:
        """
        Load ensemble models for prediction.
        
        Args:
            folder (str): Path to folder containing ensemble models
            total_models (int): Number of models in the ensemble
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create EnsembleModelClassifier instance
            self.classifier_instance = EnsembleModelClassifier(self.job_folder, self.parameters)
            
            # Load the ensemble models
            success = self.classifier_instance.load_ensemble_models(folder, total_models)
            
            if success:
                self.classifier_type = 'ensemble'
                self.is_trained = True
                self.n_classes = 2  # Ensemble models are typically binary
                print(f"Successfully loaded {total_models} ensemble models from {folder}")
            else:
                print(f"Failed to load ensemble models from {folder}")
                
            return success
            
        except Exception as e:
            print(f"Error loading ensemble models: {e}")
            return False
    
    def make_ensemble_predictions_direct(self, data: Union[np.ndarray, List], features: List[str], vote_type: str = 'soft') -> Dict[str, Any]:
        """
        Make predictions using ensemble models.
        
        Args:
            data: Input data for prediction
            features: List of feature names
            vote_type: Voting strategy ('soft', 'hard', 'majority')
            
        Returns:
            dict: Ensemble prediction results
        """
        if not self.is_trained or self.classifier_type != 'ensemble':
            raise ClassifierNotTrainedException("Ensemble models must be loaded first")
        
        return self.classifier_instance.predict_ensemble(data, features, vote_type)

    def __repr__(self) -> str:
        """String representation of the manager."""
        return f"UnifiedClassifierManager(job_folder='{self.job_folder}', trained={self.is_trained}, type={self.classifier_type})"
