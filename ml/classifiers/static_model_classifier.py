"""
Static Model Classifier

This module provides the StaticModelClassifier class for handling static models (pipeline.joblib files).
It extends the AutoMLClassifier base class to provide a consistent interface for pre-trained models.
"""

import numpy as np
import pandas as pd
from joblib import load

from .base_classifier import AutoMLClassifier
from .binary_classifier import BinaryClassifier
from .multiclass_macro_classifier import MulticlassMacroClassifier


class StaticModelClassifier(AutoMLClassifier):
    """
    Handles static models (pipeline.joblib files) by extending AutoMLClassifier.
    
    This classifier is designed for pre-trained models that are saved as joblib files,
    providing the same interface as BinaryClassifier/MulticlassClassifier but for
    models that don't need training.
    """
    
    def __init__(self, parameters, output_path='.', update_function=lambda x, y: None):
        """
        Initialize the Static Model Classifier.
        
        Args:
            parameters (dict): Configuration parameters for the classifier
            output_path (str): Path where results and models will be saved
            update_function (callable): Callback function for progress updates
        """
        super().__init__(parameters, output_path, update_function)
        self.static_model = None
        self.model_path = None
        self.n_classes = None
        self.is_binary = None
        self._delegate_classifier = None  # Will hold BinaryClassifier or MulticlassClassifier instance
    
    def load_model(self, model_path):
        """
        Load a static model from a joblib file.
        
        Args:
            model_path (str): Path to the joblib model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the model
            self.static_model = load(model_path)
            self.model_path = model_path
            
            # Try to determine if this is binary or multiclass by creating dummy data
            # and checking the output shape
            try:
                # Create a small dummy dataset to test the model
                if hasattr(self.static_model, 'n_features_in_'):
                    n_features = self.static_model.n_features_in_
                elif hasattr(self.static_model, 'feature_names_in_'):
                    n_features = len(self.static_model.feature_names_in_)
                else:
                    n_features = 10  # Default fallback
                
                dummy_data = np.random.random((1, n_features))
                
                if hasattr(self.static_model, 'predict_proba'):
                    proba_output = self.static_model.predict_proba(dummy_data)
                    self.n_classes = proba_output.shape[1]
                    self.is_binary = (self.n_classes == 2)
                else:
                    # Try to predict and see what we get
                    pred_output = self.static_model.predict(dummy_data)
                    # This is a fallback - assume binary for now
                    self.n_classes = 2
                    self.is_binary = True
                    
            except Exception as e:
                print(f"Warning: Could not determine model type automatically: {e}")
                # Default to binary
                self.n_classes = 2
                self.is_binary = True
            
            # Create appropriate delegate classifier for evaluation methods
            self._create_delegate_classifier()
            
            print(f"Loaded static model from {model_path}")
            print(f"Detected {'binary' if self.is_binary else 'multiclass'} model with {self.n_classes} classes")
            
            return True
            
        except Exception as e:
            print(f"Error loading static model from {model_path}: {e}")
            return False
    
    def _create_delegate_classifier(self):
        """
        Create a delegate classifier instance for evaluation methods.
        """
        if self.is_binary:
            self._delegate_classifier = BinaryClassifier(self.parameters, self.output_path, self.update_function)
        else:
            self._delegate_classifier = MulticlassMacroClassifier(self.parameters, self.output_path, self.update_function)
        
        # Set up the delegate classifier with our static model
        self._delegate_classifier.main_models = {'static_model': self.static_model}
    
    def predict(self, data, model_key, threshold=0.5, class_index=None):
        """
        Make predictions using the loaded static model.
        
        Args:
            data: Input data for prediction (numpy array or similar)
            model_key (str): Key identifying the model (for compatibility)
            threshold (float): Prediction threshold for binary classification
            class_index (int, optional): Specific class index for OvR-style prediction
            
        Returns:
            dict: Prediction results
        """
        if self.static_model is None:
            raise ValueError("No static model loaded. Call load_model() first.")
        
        try:
            if hasattr(self.static_model, 'predict_proba'):
                probabilities = self.static_model.predict_proba(data)
                
                if self.is_binary:
                    # Binary classification
                    predictions = (probabilities[:, 1] >= threshold).astype(int)
                    return {
                        'predicted': predictions.tolist(),
                        'probability': probabilities[:, 1].tolist(),
                        'threshold': threshold,
                        'classification_type': 'binary'
                    }
                else:
                    # Multiclass classification
                    predictions = self.static_model.predict(data)
                    
                    if class_index is not None:
                        # Return class-specific probabilities
                        class_probabilities = probabilities[:, class_index]
                        return {
                            'predicted': predictions.tolist(),
                            'probability': class_probabilities.tolist(),
                            'all_probabilities': probabilities.tolist(),
                            'class_index': class_index,
                            'classification_type': 'multiclass'
                        }
                    else:
                        # Return max probabilities (confidence in prediction)
                        max_probabilities = probabilities.max(axis=1)
                        return {
                            'predicted': predictions.tolist(),
                            'probability': max_probabilities.tolist(),
                            'all_probabilities': probabilities.tolist(),
                            'classification_type': 'multiclass'
                        }
            else:
                # Fallback for models without predict_proba
                predictions = self.static_model.predict(data)
                return {
                    'predicted': predictions.tolist(),
                    'probability': [1.0] * len(predictions),  # Default probability
                    'threshold': threshold if self.is_binary else None,
                    'classification_type': 'binary' if self.is_binary else 'multiclass'
                }
                
        except Exception as e:
            raise RuntimeError(f"Error making predictions with static model: {e}")
    
    def generalization_report(self, labels, y2, predictions, probabilities, class_index=None):
        """
        Generate generalization report for static models using delegate classifier.
        
        Args:
            labels (list): Class labels
            y2 (array): True labels
            predictions (array): Predicted labels
            probabilities (array): Prediction probabilities
            class_index (int, optional): Specific class index for OvR evaluation
            
        Returns:
            dict: Generalization metrics
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.generalization_report(labels, y2, predictions, probabilities, class_index)
    
    def get_generalization_metrics(self, data, label_column, threshold=0.5):
        """
        Get generalization metrics using delegate classifier methods.
        
        Args:
            data (dict): Test data
            label_column (str): Target column name
            threshold (float): Prediction threshold
            
        Returns:
            dict: Generalization metrics
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.get_generalization_metrics(data, 'static_model', label_column, threshold)
    
    def get_reliability_metrics(self, data, label_column):
        """
        Get reliability metrics using delegate classifier methods.
        
        Args:
            data (dict): Test data
            label_column (str): Target column name
            
        Returns:
            dict: Reliability metrics
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.get_additional_reliability(data, 'static_model', label_column)
    
    def get_precision_metrics(self, data, label_column):
        """
        Get precision-recall metrics using delegate classifier methods.
        
        Args:
            data (dict): Test data
            label_column (str): Target column name
            
        Returns:
            dict: Precision-recall metrics
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.get_additional_precision(data, 'static_model', label_column)
    
    def get_roc_metrics(self, data, label_column):
        """
        Get ROC metrics using delegate classifier methods.
        
        Args:
            data (dict): Test data
            label_column (str): Target column name
            
        Returns:
            dict: ROC metrics
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.get_additional_roc(data, 'static_model', label_column)
    
    def evaluate_generalization(self, pipeline, features, estimator, x_test, y_test, labels):
        """
        Evaluate generalization for static models using delegate classifier.
        
        Args:
            pipeline: Not used for static models
            features: Not used for static models
            estimator: Not used for static models (we use self.static_model)
            x_test (array): Test features
            y_test (array): Test labels
            labels (list): Class labels
            
        Returns:
            dict: Generalization evaluation results
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        # For static models, we can make predictions and use generalization_report
        try:
            predictions_result = self.predict(x_test, 'static_model')
            predictions = np.array(predictions_result['predicted'])
            probabilities = np.array(predictions_result['probability'])
            
            return self.generalization_report(labels, y_test, predictions, probabilities)
            
        except Exception as e:
            print(f"Error in static model generalization evaluation: {e}")
            # Fallback to empty result
            return {}
    
    def evaluate_precision_recall(self, pipeline, features, estimator, x_test, y_test):
        """
        Evaluate precision-recall using delegate classifier.
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.evaluate_precision_recall(None, None, self.static_model, x_test, y_test)
    
    def evaluate_reliability(self, pipeline, features, estimator, x_test, y_test):
        """
        Evaluate reliability using delegate classifier.
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.evaluate_reliability(None, None, self.static_model, x_test, y_test)
    
    def evaluate_roc(self, pipeline, features, estimator, x_data, y_data):
        """
        Evaluate ROC using delegate classifier.
        """
        if self._delegate_classifier is None:
            raise ValueError("No delegate classifier available. Load a model first.")
        
        return self._delegate_classifier.evaluate_roc(None, None, self.static_model, x_data, y_data)
    
    def generalize_static_model(self, payload, label, threshold=0.5):
        """
        Generalize a static model using payload data (adapted from outdated/generalization.py)
        
        Args:
            payload (dict): Data payload with 'data', 'columns', and 'features' keys
            label (str): Label column name
            threshold (float): Prediction threshold for binary classification
            
        Returns:
            dict: Generalization report results
        """
        if self.static_model is None:
            raise ValueError("No static model loaded. Call load_model() first.")
        
        # Process payload data
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        # Make predictions
        probabilities = self.static_model.predict_proba(x)
        
        if self.is_binary:
            # Binary classification
            probabilities_pos = probabilities[:, 1]
            if threshold == 0.5:
                predictions = self.static_model.predict(x)
            else:
                predictions = (probabilities_pos >= threshold).astype(int)
            
            # Generate binary labels
            labels = ['No ' + label, label]
            
            return self.generalization_report(labels, y, predictions, probabilities_pos)
        else:
            # Multiclass classification
            predictions = self.static_model.predict(x)
            
            # Generate multiclass labels
            unique_labels = sorted(y.unique())
            labels = [f'Class {int(cls)}' for cls in unique_labels]
            
            return self.generalization_report(labels, y, predictions, probabilities)

    def fit(self, x_train, x_val, y_train, y_val, x_test, y_test, feature_names, labels):
        """
        Fit method for compatibility with AutoMLClassifier interface.
        
        For static models, this method doesn't actually train anything since the model
        is already pre-trained. It just validates that a model has been loaded.
        
        Args:
            x_train (array): Training features (not used)
            x_val (array): Validation features (not used)
            y_train (array): Training labels (not used)
            y_val (array): Validation labels (not used)
            x_test (array): Test features (not used)
            y_test (array): Test labels (not used)
            feature_names (list): List of feature names (not used)
            labels (list): List of class labels (not used)
            
        Returns:
            bool: True if a static model is loaded, False otherwise
        """
        if self.static_model is None:
            print("Error: No static model loaded. Call load_model() first.")
            return False
        
        print("Static model is ready for predictions.")
        return True
