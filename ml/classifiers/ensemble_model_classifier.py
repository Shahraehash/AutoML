"""
Ensemble Model Classifier for handling multiple model combinations
"""

import json
import joblib
import pandas as pd
import numpy as np
from .base_classifier import AutoMLClassifier


class EnsembleModelClassifier(AutoMLClassifier):
    """
    Classifier for handling ensemble models (multiple model combinations)
    
    Ensemble models use multiple trained models and combine their predictions
    using different voting strategies (soft voting, hard voting, etc.)
    """
    
    def __init__(self, job_folder, parameters=None):
        super().__init__(job_folder, parameters)
        self.models = []
        self.model_features = []
        self.total_models = 0
        self.ensemble_config = None
        self.is_loaded = False
    
    def load_ensemble_models(self, folder, total_models):
        """
        Load ensemble models from the job folder
        
        Args:
            folder (str): Path to job folder containing ensemble models
            total_models (int): Number of models in the ensemble
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            self.total_models = total_models
            self.models = []
            self.model_features = []
            
            # Load each ensemble model
            for i in range(total_models):
                # Load model
                model_path = folder + f'/ensemble{i}.joblib'
                model = joblib.load(model_path)
                self.models.append(model)
                
                # Load features for this model
                features_path = folder + f'/ensemble{i}_features.json'
                with open(features_path) as feature_file:
                    features = json.load(feature_file)
                    self.model_features.append(features)
            
            # Load ensemble configuration
            ensemble_config_path = folder + '/ensemble.json'
            with open(ensemble_config_path) as config_file:
                self.ensemble_config = json.load(config_file)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading ensemble models: {e}")
            self.is_loaded = False
            return False
    
    def predict_ensemble(self, data, features, vote_type='soft'):
        """
        Make ensemble predictions using multiple models
        
        Args:
            data: Input data for prediction
            features: List of feature names
            vote_type: Voting strategy ('soft', 'hard', 'majority')
            
        Returns:
            dict: Prediction results with 'predicted' and 'probability' lists
        """
        if not self.is_loaded:
            raise Exception("Ensemble models not loaded")
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=features)
        
        # Get predictions from each model
        all_predictions = []
        all_probabilities = []
        
        for i, (model, model_features) in enumerate(zip(self.models, self.model_features)):
            # Extract features for this model
            model_data = df[model_features].to_numpy()
            
            # Get predictions
            predictions = model.predict(model_data)
            all_predictions.append(predictions)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(model_data)[:, 1]
            else:
                probabilities = predictions
            all_probabilities.append(probabilities)
        
        # Convert to numpy arrays for easier manipulation
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Apply voting strategy
        final_predictions = []
        final_probabilities = []
        
        if vote_type == 'soft':
            # Soft voting: average probabilities, then threshold
            avg_probabilities = np.mean(all_probabilities, axis=0)
            final_predictions = (avg_probabilities > 0.5).astype(int)
            final_probabilities = avg_probabilities
            
        elif vote_type == 'hard' or vote_type == 'majority':
            # Hard/majority voting: majority vote on predictions
            for i in range(all_predictions.shape[1]):
                sample_predictions = all_predictions[:, i]
                # Count positive predictions
                positive_votes = np.sum(sample_predictions > 0)
                negative_votes = np.sum(sample_predictions <= 0)
                
                if positive_votes > negative_votes:
                    final_predictions.append(1)
                else:
                    final_predictions.append(0)
                
                # For probability, use average of probabilities
                final_probabilities.append(np.mean(all_probabilities[:, i]))
        
        else:
            # Default to soft voting
            avg_probabilities = np.mean(all_probabilities, axis=0)
            final_predictions = (avg_probabilities > 0.5).astype(int)
            final_probabilities = avg_probabilities
        
        return {
            'predicted': final_predictions.tolist() if hasattr(final_predictions, 'tolist') else final_predictions,
            'probability': final_probabilities.tolist() if hasattr(final_probabilities, 'tolist') else final_probabilities
        }
    
    def predict(self, data, model_key=None, threshold=0.5, vote_type='soft'):
        """
        Standard predict interface for compatibility
        
        Args:
            data: Input data
            model_key: Not used for ensemble models
            threshold: Not used for ensemble models
            vote_type: Voting strategy for ensemble
            
        Returns:
            dict: Prediction results
        """
        # Extract features from data structure
        if isinstance(data, dict) and 'data' in data and 'features' in data:
            return self.predict_ensemble(data['data'], data['features'], vote_type)
        else:
            raise ValueError("Data must be a dict with 'data' and 'features' keys")
    
    def generalize_ensemble_models(self, dataset_folder, label):
        """
        Generalize ensemble models using test data (adapted from outdated/generalization.py)
        
        Args:
            dataset_folder (str): Path to dataset folder containing test.csv
            label (str): Label column name
            
        Returns:
            dict: Ensemble generalization results with both soft and hard voting
        """
        from ..import_data import import_csv
        
        if not self.is_loaded:
            raise Exception("Ensemble models not loaded")
        
        # Import test data
        x_test, y_test, feature_names, _, _ = import_csv(dataset_folder + '/test.csv', label)
        
        # Convert to DataFrame for easier feature selection
        data = pd.DataFrame(x_test, columns=feature_names)
        
        # Get predictions using both voting methods
        soft_result = self.predict_ensemble(data.values, feature_names, 'soft')
        hard_result = self.predict_ensemble(data.values, feature_names, 'hard')
        
        # Determine if this is binary or multi-class
        unique_labels = sorted(y_test.unique())
        if len(unique_labels) == 2:
            labels = ['No ' + label, label]
        else:
            labels = [f'Class {int(cls)}' for cls in unique_labels]

        # Generate generalization reports for both voting methods
        soft_generalization = self._generate_ensemble_generalization_report(
            labels, y_test, soft_result['predicted'], soft_result['probability']
        )
        
        hard_generalization = self._generate_ensemble_generalization_report(
            labels, y_test, hard_result['predicted'], hard_result['probability']
        )
        
        return {
            'soft_generalization': soft_generalization,
            'hard_generalization': hard_generalization
        }
    
    def _generate_ensemble_generalization_report(self, labels, y_test, predictions, probabilities):
        """
        Generate generalization report for ensemble predictions
        
        Args:
            labels (list): Class labels
            y_test (array): True labels
            predictions (list): Predicted labels
            probabilities (list): Prediction probabilities
            
        Returns:
            dict: Generalization metrics
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, roc_curve
        from ..stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Get unique classes from actual data
        unique_classes = sorted(np.unique(y_test))
        n_classes = len(unique_classes)
        
        # If labels are provided but don't match the number of classes, generate appropriate labels
        if labels is None or len(labels) != n_classes:
            if n_classes == 2:
                labels = ['Class 0', 'Class 1']
            else:
                labels = [f'Class {int(cls)}' for cls in unique_classes]
        
        print('\t', classification_report(y_test, predictions, target_names=labels).replace('\n', '\n\t'))
        print('\tGeneralization:')
        
        accuracy = accuracy_score(y_test, predictions)
        print('\t\tAccuracy:', accuracy)
        
        if n_classes == 2:
            # Binary classification metrics
            auc = roc_auc_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, probabilities)
            print('\t\tROC AUC:', roc_auc)
            
            _, tpr, _ = roc_curve(y_test, probabilities)
            
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                
            mcc = matthews_corrcoef(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            sensitivity = tp / (tp+fn)
            specificity = tn / (tn+fp)
            prevalence = (tp + fn) / (len(y_test))

            return {
                'accuracy': round(accuracy, 4),
                'acc_95_ci': clopper_pearson(tp+tn, len(y_test)),
                'mcc': round(mcc, 4),
                'avg_sn_sp': round(auc, 4),
                'roc_auc': round(roc_auc, 4),
                'roc_auc_95_ci': roc_auc_ci(roc_auc, tpr),
                'f1': round(f1, 4),
                'sensitivity': round(sensitivity, 4),
                'sn_95_ci': clopper_pearson(tp, tp+fn),
                'specificity': round(specificity, 4),
                'sp_95_ci': clopper_pearson(tn, tn+fp),
                'prevalence': round(prevalence, 4),
                'pr_95_ci': clopper_pearson(tp+fn, len(y_test)),
                'ppv': round(tp / (tp+fp), 4) if tp+fp > 0 else 0,
                'ppv_95_ci': ppv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
                'npv': round(tn / (tn+fn), 4) if tn+fn > 0 else 0,
                'npv_95_ci': npv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
                'tn': int(tn),
                'tp': int(tp),
                'fn': int(fn),
                'fp': int(fp)
            }
        else:
            # Multiclass classification metrics (macro-averaged)
            # Convert probabilities to proper format for multiclass ROC AUC
            if probabilities.ndim == 1:
                # If probabilities is 1D, we need to create a proper probability matrix
                # This is a fallback - ideally ensemble should return proper multiclass probabilities
                prob_matrix = np.zeros((len(predictions), n_classes))
                for i, pred in enumerate(predictions):
                    prob_matrix[i, pred] = probabilities[i]
                probabilities = prob_matrix
            
            roc_auc_macro = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
            print('\t\tROC AUC (macro):', roc_auc_macro)
            
            # Calculate confusion matrix for OvR decomposition
            cnf_matrix = confusion_matrix(y_test, predictions)
            
            # Calculate per-class metrics for averaging
            all_sensitivities = []
            all_specificities = []
            all_prevalences = []
            all_f1_scores = []
            all_mcc_scores = []
            
            # Calculate confusion matrix components for macro averaging
            fp_all = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
            fn_all = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
            tp_all = (np.diag(cnf_matrix)).astype(float)
            tn_all = (cnf_matrix.sum() - (fp_all + fn_all + tp_all)).astype(float)
            
            # Get TPR for ROC AUC CI calculation
            tpr_all = tp_all / (tp_all + fn_all)
            
            for class_idx in range(n_classes):
                tp = tp_all[class_idx]
                tn = tn_all[class_idx]
                fp = fp_all[class_idx]
                fn = fn_all[class_idx]
                
                # Calculate metrics for this class
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                prevalence = (tp + fn) / len(y_test)
                
                all_sensitivities.append(sensitivity)
                all_specificities.append(specificity)
                all_prevalences.append(prevalence)
                
                # Class-specific F1 and MCC for averaging
                f1_class = f1_score(y_test, predictions, labels=[class_idx], average=None)[0]
                all_f1_scores.append(f1_class)
                
                y_binary = (y_test == class_idx).astype(int)
                pred_binary = (predictions == class_idx).astype(int)
                mcc_class = matthews_corrcoef(y_binary, pred_binary)
                all_mcc_scores.append(mcc_class)
                
            # Macro-averaged metrics
            macro_sensitivity = np.mean(all_sensitivities)
            macro_specificity = np.mean(all_specificities)
            macro_prevalence = np.mean(all_prevalences)
            macro_f1 = np.mean(all_f1_scores)
            macro_mcc = np.mean(all_mcc_scores)
            
            # Sum across classes for confusion matrix totals (for CI calculations)
            tp_sum = int(np.sum(tp_all))
            tn_sum = int(np.sum(tn_all))
            fp_sum = int(np.sum(fp_all))
            fn_sum = int(np.sum(fn_all))
            
            return {
                'accuracy': round(accuracy, 4),
                'acc_95_ci': clopper_pearson(tp_sum+tn_sum, len(y_test)),
                'mcc': round(macro_mcc, 4),
                'avg_sn_sp': round((macro_sensitivity + macro_specificity) / 2, 4),
                'roc_auc': round(roc_auc_macro, 4),
                'roc_auc_95_ci': roc_auc_ci(roc_auc_macro, tpr_all),
                'f1': round(macro_f1, 4),
                'sensitivity': round(macro_sensitivity, 4),
                'sn_95_ci': clopper_pearson(tp_sum, tp_sum+fn_sum) if tp_sum+fn_sum > 0 else (0, 0),
                'specificity': round(macro_specificity, 4),
                'sp_95_ci': clopper_pearson(tn_sum, tn_sum+fp_sum) if tn_sum+fp_sum > 0 else (0, 0),
                'prevalence': round(macro_prevalence, 4),
                'pr_95_ci': clopper_pearson(tp_sum+fn_sum, len(y_test)),
                'ppv': round(tp_sum / (tp_sum+fp_sum), 4) if tp_sum+fp_sum > 0 else 0,
                'ppv_95_ci': ppv_95_ci(macro_sensitivity, macro_specificity, tp_sum+fn_sum, fp_sum+tn_sum, macro_prevalence),
                'npv': round(tn_sum / (tn_sum+fn_sum), 4) if tn_sum+fn_sum > 0 else 0,
                'npv_95_ci': npv_95_ci(macro_sensitivity, macro_specificity, tp_sum+fn_sum, fp_sum+tn_sum, macro_prevalence),
                'tn': tn_sum,
                'tp': tp_sum,
                'fn': fn_sum,
                'fp': fp_sum
            }

    def get_model_info(self):
        """
        Get information about the loaded ensemble models
        
        Returns:
            dict: Model information
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        model_types = [type(model).__name__ for model in self.models]
        
        return {
            "status": "loaded",
            "model_type": "ensemble",
            "total_models": self.total_models,
            "model_types": model_types,
            "model_features": self.model_features,
            "ensemble_config": self.ensemble_config
        }
