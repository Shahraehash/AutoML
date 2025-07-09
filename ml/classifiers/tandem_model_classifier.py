"""
Tandem Model Classifier for handling NPV + PPV model combinations
"""

import json
import joblib
import pandas as pd
import numpy as np
from .base_classifier import AutoMLClassifier


class TandemModelClassifier(AutoMLClassifier):
    """
    Classifier for handling tandem models (NPV + PPV combinations)
    
    Tandem models use two separate models:
    1. NPV (Negative Predictive Value) model - primary model
    2. PPV (Positive Predictive Value) model - override model for positive predictions
    """
    
    def __init__(self, job_folder, parameters=None):
        super().__init__(job_folder, parameters)
        self.npv_model = None
        self.ppv_model = None
        self.npv_features = None
        self.ppv_features = None
        self.is_loaded = False
    
    def load_tandem_models(self, folder):
        """
        Load NPV and PPV models from the job folder
        
        Args:
            folder (str): Path to job folder containing tandem models
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # Load NPV model
            npv_model_path = folder + '/tandem_npv.joblib'
            self.npv_model = joblib.load(npv_model_path)
            
            # Load PPV model
            ppv_model_path = folder + '/tandem_ppv.joblib'
            self.ppv_model = joblib.load(ppv_model_path)
            
            # Load NPV features
            with open(folder + '/tandem_npv_features.json') as feature_file:
                self.npv_features = json.load(feature_file)
            
            # Load PPV features
            with open(folder + '/tandem_ppv_features.json') as feature_file:
                self.ppv_features = json.load(feature_file)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading tandem models: {e}")
            self.is_loaded = False
            return False
    
    def predict_tandem(self, data, features):
        """
        Make tandem predictions using NPV and PPV models
        
        Args:
            data: Input data for prediction
            features: List of feature names
            
        Returns:
            dict: Prediction results with 'predicted' and 'probability' lists
        """
        if not self.is_loaded:
            raise Exception("Tandem models not loaded")
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=features)
        
        # Get NPV predictions
        npv_data = df[self.npv_features].to_numpy()
        npv_predictions = self.npv_model.predict(npv_data)
        npv_probabilities = self.npv_model.predict_proba(npv_data)[:, 1] if hasattr(self.npv_model, 'predict_proba') else npv_predictions
        
        # Get PPV predictions
        ppv_data = df[self.ppv_features].to_numpy()
        ppv_predictions = self.ppv_model.predict(ppv_data)
        ppv_probabilities = self.ppv_model.predict_proba(ppv_data)[:, 1] if hasattr(self.ppv_model, 'predict_proba') else ppv_predictions
        
        # Apply tandem logic
        final_predictions = []
        final_probabilities = []
        
        for i in range(len(npv_predictions)):
            npv_pred = npv_predictions[i]
            npv_prob = npv_probabilities[i]
            ppv_pred = ppv_predictions[i]
            ppv_prob = ppv_probabilities[i]
            
            # PPV override logic: if PPV predicts negative, set to -1
            if ppv_pred <= 0:
                ppv_pred = -1
            
            # If NPV predicts positive, use PPV prediction and probability
            if npv_pred > 0:
                final_predictions.append(ppv_pred)
                final_probabilities.append(ppv_prob)
            else:
                final_predictions.append(npv_pred)
                final_probabilities.append(npv_prob)
        
        return {
            'predicted': final_predictions,
            'probability': final_probabilities
        }
    
    def predict(self, data, model_key=None, threshold=0.5):
        """
        Standard predict interface for compatibility
        
        Args:
            data: Input data
            model_key: Not used for tandem models
            threshold: Not used for tandem models
            
        Returns:
            dict: Prediction results
        """
        # Extract features from data structure
        if isinstance(data, dict) and 'data' in data and 'features' in data:
            return self.predict_tandem(data['data'], data['features'])
        else:
            raise ValueError("Data must be a dict with 'data' and 'features' keys")
    
    def get_model_info(self):
        """
        Get information about the loaded tandem models
        
        Returns:
            dict: Model information
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": "tandem",
            "npv_features": self.npv_features,
            "ppv_features": self.ppv_features,
            "npv_model_type": type(self.npv_model).__name__,
            "ppv_model_type": type(self.ppv_model).__name__
        }
