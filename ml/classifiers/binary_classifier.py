"""
Binary Classification Classifier

This module provides the BinaryClassifier class for handling binary classification tasks.
It extends the AutoMLClassifier base class with binary-specific logic.
"""

import numpy as np
import pandas as pd
import json
import os
from timeit import default_timer as timer
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve
from nyoka import skl_to_pmml, xgboost_to_pmml


from .base_classifier import AutoMLClassifier
from ..preprocess import preprocess
from ..utils import model_key_to_name, decimate_points
from ..stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci


class BinaryClassifier(AutoMLClassifier):
    """
    Handles binary classification tasks.
    
    This classifier is optimized for datasets with exactly 2 classes and provides
    binary-specific evaluation metrics and model handling.
    """
    
    def __init__(self, parameters, output_path='.', update_function=lambda x, y: None):
        """
        Initialize the Binary Classifier.
        
        Args:
            parameters (dict): Configuration parameters for the classifier
            output_path (str): Path where results and models will be saved
            update_function (callable): Callback function for progress updates
        """
        super().__init__(parameters, output_path, update_function)
    
    def evaluate_precision_recall(self, pipeline, features, estimator, x_test, y_test):
        """
        Evaluate precision-recall for binary classification.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            
        Returns:
            dict: Precision-recall evaluation results
        """
        # Transform values based on the pipeline
        x_test = preprocess(features, pipeline, x_test)
        
        if hasattr(estimator, 'decision_function'):
            scores = estimator.decision_function(x_test)
            # For binary classification, decision_function returns 1D array
            if scores.ndim == 1:
                precision, recall, _ = precision_recall_curve(y_test, scores)
            else:
                precision, recall, _ = precision_recall_curve(y_test, scores[:, 1])
        else:
            # Use predict_proba
            probabilities = estimator.predict_proba(x_test)
            precision, recall, _ = precision_recall_curve(y_test, probabilities[:, 1])

        # Apply decimation
        recall, precision = decimate_points(
            [round(num, 4) for num in list(recall)],
            [round(num, 4) for num in list(precision)]
        )

        return {
            'precision': list(precision),
            'recall': list(recall)
        }
    
    def evaluate_reliability(self, pipeline, features, estimator, x_test, y_test):
        """
        Evaluate reliability for binary classification.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            
        Returns:
            dict: Reliability evaluation results
        """
        # Transform values based on the pipeline
        x_test = preprocess(features, pipeline, x_test)

        if hasattr(estimator, 'decision_function'):
            probabilities = estimator.decision_function(x_test)
            # Binary Classification
            if np.count_nonzero(probabilities):
                if probabilities.max() - probabilities.min() == 0:
                    probabilities = [0] * len(probabilities)
                else:
                    probabilities = (probabilities - probabilities.min()) / \
                        (probabilities.max() - probabilities.min())
            fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')
            brier_score = brier_score_loss(y_test, probabilities)
        else:
            # Binary classification
            probabilities = estimator.predict_proba(x_test)[:, 1]
            fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')
            brier_score = brier_score_loss(y_test, probabilities)

        return {
            'brier_score': round(brier_score, 4),
            'fop': [round(num, 4) for num in list(fop)],
            'mpv': [round(num, 4) for num in list(mpv)]
        }
    
    def evaluate_roc(self, pipeline, features, estimator, x_data, y_data):
        """
        Evaluate ROC for binary classification.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_data (array): Features for ROC evaluation
            y_data (array): Labels for ROC evaluation
            
        Returns:
            dict: ROC evaluation results
        """
        # Transform values based on the pipeline
        x_data = preprocess(features, pipeline, x_data)

        probabilities = estimator.predict_proba(x_data)
        
        # Binary classification
        fpr, tpr, _ = roc_curve(y_data, probabilities[:, 1])
        roc_auc = roc_auc_score(y_data, probabilities[:, 1])
        
        fpr, tpr = decimate_points(
            [round(num, 4) for num in list(fpr)],
            [round(num, 4) for num in list(tpr)]
        )

        return {
            'fpr': list(fpr),
            'tpr': list(tpr),
            'roc_auc': roc_auc
        }
    
    def predict(self, data, model_key, threshold=0.5):
        """
        Make predictions using a specific trained binary model.
        
        Args:
            data: Input data for prediction (numpy array or similar)
            model_key (str): Key identifying the model
            threshold (float): Prediction threshold for binary classification
            
        Returns:
            dict: Prediction results with binary-specific logic
        """
        if model_key not in self.main_models:
            raise KeyError(f"Model {model_key} not found")
        
        model = self.main_models[model_key]
        
        # Binary classification prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)
            # For binary classification, use threshold on positive class probability
            predictions = (probabilities[:, 1] >= threshold).astype(int)
            return {
                'predicted': predictions.tolist(),
                'probability': probabilities[:, 1].tolist(),
                'threshold': threshold,
                'classification_type': 'binary'
            }
        else:
            # Fallback for models without predict_proba
            predictions = model.predict(data)
            return {
                'predicted': predictions.tolist(),
                'probability': [1.0] * len(predictions),  # Default probability
                'threshold': threshold,
                'classification_type': 'binary'
            }
    
    def evaluate_generalization(self, pipeline, features, estimator, x_test, y_test, labels):
        """
        Evaluate generalization for binary classification.
        
        Args:
            pipeline: Sklearn pipeline
            features: Feature information
            estimator: Trained estimator
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            labels (list): Class labels
            
        Returns:
            dict: Generalization evaluation results
        """
        # Process test data based on pipeline
        x_test = preprocess(features, pipeline, x_test)
        proba = estimator.predict_proba(x_test)
        
        # Binary classification
        probabilities = proba[:, 1]
        predictions = estimator.predict(x_test)
        
        return self.generalization_report(labels, y_test, predictions, probabilities)
    

    def generalization_report(self, labels, y2, predictions, probabilities, class_index=None):
        """
        Generate generalization report for binary classification.
        
        Args:
            labels (list): Class labels
            y2 (array): True labels
            predictions (array): Predicted labels
            probabilities (array): Prediction probabilities
            class_index (int, optional): Not used in binary classification
            
        Returns:
            dict: Binary classification generalization metrics
        """
        # Generate labels if not provided correctly for binary classification
        if labels is None or len(labels) != 2:
            labels = ['Class 0', 'Class 1']
        
        print('\t', classification_report(y2, predictions, target_names=labels).replace('\n', '\n\t'))
        print('\tGeneralization:')
        
        accuracy = accuracy_score(y2, predictions)
        print('\t\tAccuracy:', accuracy)

        auc = roc_auc_score(y2, predictions)
        roc_auc = roc_auc_score(y2, probabilities)
        print('\t\tROC AUC:', roc_auc)
        
        _, tpr, _ = roc_curve(y2, probabilities)
        
        tn, fp, fn, tp = confusion_matrix(y2, predictions).ravel()
            
        mcc = matthews_corrcoef(y2, predictions)
        f1 = f1_score(y2, predictions)

        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        prevalence = (tp + fn) / (len(y2))

        return {
            'accuracy': round(accuracy, 4),
            'acc_95_ci': clopper_pearson(tp+tn, len(y2)),
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
            'pr_95_ci': clopper_pearson(tp+fn, len(y2)),
            'ppv': round(tp / (tp+fp), 4) if tp+fp > 0 else 0,
            'ppv_95_ci': ppv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
            'npv': round(tn / (tn+fn), 4) if tn+fn > 0 else 0,
            'npv_95_ci': npv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
            'tn': int(tn),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp)
        }
    
    def create_model(self, key, hyper_parameters, selected_features, dataset_path=None, label_column=None, output_path='.', threshold=.5):
        """
        Create and export a binary classification model (adapted from outdated/create_model.py)
        
        Args:
            key (str): Model key identifying the pipeline configuration
            hyper_parameters (dict): Hyperparameters for the model
            selected_features (list): List of selected feature names
            dataset_path (str): Path to dataset folder containing train.csv and test.csv
            label_column (str): Name of the target column
            output_path (str): Path where outputs will be saved
            threshold (float): Prediction threshold for binary classification
            
        Returns:
            dict: Generalization results for the created model
        """        
        if dataset_path is None:
            print('Missing dataset path')
            return {}

        if label_column is None:
            print('Missing column name for classifier target')
            return {}

        # Import data
        (x_train, _, y_train, _, x2, y2, features, _) = \
            import_data(dataset_path + '/train.csv', dataset_path + '/test.csv', label_column)

        # Validate binary classification
        n_classes = len(np.unique(y_train))
        if n_classes != 2:
            raise ValueError(f"BinaryClassifier.create_model expects 2 classes, got {n_classes}")

        # Get pipeline details from the key
        scaler, feature_selector, estimator, _, _ = explode_key(key)
        steps = []

        # Drop the unused features
        if 'pca-' not in feature_selector:
            for index, feature in reversed(list(enumerate(features))):
                if feature not in selected_features:
                    x_train = np.delete(x_train, index, axis=1)
                    x2 = np.delete(x2, index, axis=1)

        # Add the scaler, if used
        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        # Add the feature transformer
        if 'pca-' in feature_selector:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        # Add the estimator with proper XGBoost configuration
        if estimator == 'gb':
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

        # Binary classification labels
        labels = ['No ' + label_column, label_column]

        # Assess the model performance and store the results
        generalization_result = self.evaluate_generalization(pipeline, model['features'], pipeline['estimator'], x2, y2, labels)
        with open(output_path + '/pipeline.json', 'w') as statsfile:
            json.dump(generalization_result, statsfile)

        # Dump the pipeline to a file
        dump(pipeline, output_path + '/pipeline.joblib')
        pd.DataFrame([selected_features]).to_csv(output_path + '/input.csv', index=False, header=False)

        # Export the model as a PMML
        try:
            if estimator == 'gb':
                if xgboost_to_pmml:
                    xgboost_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
            else:
                if skl_to_pmml:
                    skl_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
        except Exception:
            try:
                os.remove(output_path + '/pipeline.pmml')
            except OSError:
                pass

        return generalization_result

    def fit(self, x_train, x_val, y_train, y_val, x_test, y_test, feature_names, labels):
        """
        Train binary classification models.
        
        Args:
            x_train (array): Training features (for model fitting)
            x_val (array): Validation features (for hyperparameter tuning)
            y_train (array): Training labels (for model fitting)
            y_val (array): Validation labels (for hyperparameter tuning)
            x_test (array): Test features (for final generalization evaluation)
            y_test (array): Test labels (for final generalization evaluation)
            feature_names (list): List of feature names
            labels (list): List of class labels
            
        Returns:
            bool: True if successful, False otherwise
        """
        start = timer()
        n_classes = len(np.unique(y_train))
        
        # Validate that this is indeed binary classification
        if n_classes != 2:
            raise ValueError(f"BinaryClassifier expects 2 classes, got {n_classes}")
        
        # Initialize reports
        self.initialize_reports()
        
        # Generate all pipeline combinations
        all_pipelines = self.generate_pipeline_combinations()
        
        print(f"Starting binary classification with {len(all_pipelines)} pipeline combinations...")
        
        for index, (estimator, scaler, feature_selector, searcher) in enumerate(all_pipelines):
            
            # Trigger callback for task monitoring
            self.update_function(index, len(all_pipelines))
            
            key = '__'.join([scaler, feature_selector, estimator, searcher])
            print('Generating ' + model_key_to_name(key))
            
            # Generate the pipeline
            pipeline_result = self.create_pipeline(estimator, scaler, feature_selector, searcher, y_train)
            pipeline = pipeline_result[0]
            pipeline_fits = pipeline_result[1]
            
            # Track total fits
            if estimator not in self.total_fits:
                self.total_fits[estimator] = 0
            self.total_fits[estimator] += pipeline_fits
            
            # Fit the pipeline
            model = self.train_model(pipeline, feature_names, x_train, y_train)
            self.performance_report_writer.writerow([key, model['train_time']])
            
            # Process each scorer
            for scorer in self.scorers:
                scorer_key = key + '__' + scorer
                candidates = self.refit_candidates(pipeline, model['features'], estimator, scorer, x_train, y_train)
                self.total_fits[estimator] += len(candidates)
                
                # Process each candidate
                for position, candidate in enumerate(candidates):
                    print('\t#%d' % (position+1))
                    
                    # Create base result
                    result = self.create_base_result(
                        scorer_key, estimator, scaler, feature_selector, searcher, scorer, n_classes, position
                    )
                    
                    # Store main model
                    self.main_models[result['key']] = candidate['best_estimator']
                    
                    # Evaluate the model using the base class method
                    evaluation_result = self.evaluate_model_complete(
                        pipeline, model['features'], candidate['best_estimator'],
                        x_val, y_val, x_test, y_test, labels
                    )
                    
                    # Update result with evaluation metrics
                    result.update(evaluation_result)
                    result.update({
                        'selected_features': list(model['selected_features']),
                        'feature_scores': model['feature_scores'],
                        'best_params': candidate['best_params']
                    })
                    
                    # Write result to CSV
                    self.write_result_to_csv(result)
        
        # Finalize reports and save results
        self.finalize_reports(start, n_classes)
        
        print(f"Binary classification completed successfully!")
        print(f"Generated {len(self.main_models)} models")
        
        return True
