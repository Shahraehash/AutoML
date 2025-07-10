"""
One-vs-Rest (OvR) Classifier

This module provides the OvRClassifier class for handling One-vs-Rest classification tasks.
It extends the AutoMLClassifier base class with OvR-specific logic, creating separate
binary classifiers for each class.
"""

import numpy as np
from timeit import default_timer as timer

from .multiclass_macro_classifier import MulticlassMacroClassifier
from .utils.utils import model_key_to_name


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
    
    def predict(self, data, model_key, class_index=None):
        """
        Make predictions using OvR models.
        
        Args:
            data: Input data for prediction (numpy array or similar)
            model_key (str): Key identifying the model
            class_index (int, optional): Specific class index for OvR model
            
        Returns:
            dict: Prediction results with OvR-specific logic
        """
        if class_index is not None:
            # Use specific OvR model for this class (re-optimized binary classifier)
            ovr_key = f"{model_key}_ovr_class_{class_index}"
            if ovr_key in self.ovr_models:
                model = self.ovr_models[ovr_key]
                
                # OvR model prediction (binary for this class vs rest)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(data)
                    # For OvR, this is binary classification for class vs rest
                    predictions = model.predict(data)
                    return {
                        'predicted': predictions.tolist(),
                        'probability': probabilities[:, 1].tolist(),  # Probability of being this class
                        'class_index': class_index,
                        'classification_type': 'ovr_binary',
                        'model_type': 'reoptimized_ovr' if self.reoptimize_ovr else 'main_model_ovr'
                    }
                else:
                    predictions = model.predict(data)
                    return {
                        'predicted': predictions.tolist(),
                        'probability': [1.0] * len(predictions),
                        'class_index': class_index,
                        'classification_type': 'ovr_binary',
                        'model_type': 'reoptimized_ovr' if self.reoptimize_ovr else 'main_model_ovr'
                    }
            else:
                raise KeyError(f"OvR model {ovr_key} not found")
        
        else:
            # Use main multiclass model (fallback to parent class behavior)
            if model_key not in self.main_models:
                raise KeyError(f"Model {model_key} not found")
            
            model = self.main_models[model_key]
            
            # Multiclass prediction using main model
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data)
                predictions = model.predict(data)
                
                # Return max probabilities (confidence in prediction)
                max_probabilities = probabilities.max(axis=1)
                return {
                    'predicted': predictions.tolist(),
                    'probability': max_probabilities.tolist(),
                    'all_probabilities': probabilities.tolist(),
                    'classification_type': 'ovr_multiclass',
                    'model_type': 'main_model'
                }
            else:
                predictions = model.predict(data)
                return {
                    'predicted': predictions.tolist(),
                    'probability': [1.0] * len(predictions),
                    'classification_type': 'ovr_multiclass',
                    'model_type': 'main_model'
                }
    
    def predict_all_ovr_classes(self, data, model_key):
        """
        Make predictions using all OvR models and aggregate results.
        
        Args:
            data: Input data for prediction
            model_key (str): Key identifying the base model
            
        Returns:
            dict: Aggregated OvR predictions across all classes
        """
        n_classes = len(np.unique(list(range(self.metadata.get('n_classes', 3)))))  # Get from metadata
        
        class_probabilities = []
        class_predictions = []
        
        for class_idx in range(n_classes):
            try:
                result = self.predict(data, model_key, class_index=class_idx)
                class_probabilities.append(result['probability'])
                class_predictions.append(result['predicted'])
            except KeyError:
                # If OvR model doesn't exist, skip this class
                continue
        
        if not class_probabilities:
            raise ValueError("No OvR models found for aggregation")
        
        # Convert to numpy arrays for easier manipulation
        class_probs_array = np.array(class_probabilities).T  # Shape: (n_samples, n_classes)
        
        # Aggregate predictions: choose class with highest probability
        final_predictions = np.argmax(class_probs_array, axis=1)
        max_probabilities = np.max(class_probs_array, axis=1)
        
        return {
            'predicted': final_predictions.tolist(),
            'probability': max_probabilities.tolist(),
            'all_class_probabilities': class_probs_array.tolist(),
            'classification_type': 'ovr_aggregated',
            'model_type': 'reoptimized_ovr' if self.reoptimize_ovr else 'main_model_ovr',
            'n_classes_used': len(class_probabilities)
        }
    

    def _generate_ovr_models_and_results(self, pipeline, features, main_model, main_result, 
                                        x_train, y_train, x_test, y_test, x2, y2, labels,
                                        estimator, scorer, reoptimize_ovr=False):
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
            y_test_binary = (y_test == actual_class_value).astype(int)
            y2_binary = (y2 == actual_class_value).astype(int)
            
            
            # Check if we have samples from both classes (class vs rest)
            unique_classes_in_binary = np.unique(y_binary)
            
            if len(unique_classes_in_binary) < 2:
                # Only one class present - cannot train binary classifier
                print(f"\t\t\tWarning: Class {class_idx} vs rest has only one class present ({unique_classes_in_binary})")
                print(f"\t\t\tSkipping re-optimization for class {class_idx}, using main model instead")
                
                # Fall back to efficient mode (use main model)
                ovr_model = main_model
                ovr_best_params = main_result['best_params']
                
                # Use multiclass classification path with class_idx
                class_metrics = self._compute_class_specific_results(
                    pipeline, features, ovr_model, x2, y2, n_classes, 
                    main_result['key'], class_idx, x_train, y_train, x_test, y_test
                )
            else:
                # Re-optimization mode: Train actual OvR model
                try:
                    ovr_candidates = self.refit_candidates(
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
                    class_metrics = self._compute_binary_class_results(
                        pipeline, features, ovr_model, 
                        x2, y2_binary,           # Binary generalization data
                        x_train, y_binary,       # Binary training data
                        x_test, y_test_binary    # Binary test data
                    )
                except Exception as e:
                    # Handle any training errors (e.g., XGBoost num_class error)
                    print(f"\t\t\tError training OvR model for class {class_idx}: {str(e)}")
                    print(f"\t\t\tFalling back to main model for class {class_idx}")
                    
                    # Fall back to efficient mode (use main model)
                    ovr_model = main_model
                    ovr_best_params = main_result['best_params']
                    
                    # Use multiclass classification path with class_idx
                    class_metrics = self._compute_class_specific_results(
                        pipeline, features, ovr_model, x2, y2, n_classes, 
                        main_result['key'], class_idx, x_train, y_train, x_test, y_test
                    )
                            
            # Store class data for .pkl.gz file
            all_class_data['class_data'][class_idx] = class_metrics
            
            # Create CSV entry for this OvR model using already computed metrics
            csv_entry = self._create_ovr_csv_entry(
                main_result, class_idx, ovr_best_params, class_metrics
            )
            
            csv_entries.append(csv_entry)
        
        return csv_entries, all_class_data, ovr_models, total_fits

    def fit(self, x_train, x_val, y_train, y_val, x_test, y_test, feature_names, labels):
        """
        Train OvR classification models.
        
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
        
        # Validate that this is indeed multiclass classification
        if n_classes <= 2:
            raise ValueError(f"OvRClassifier expects more than 2 classes, got {n_classes}")
        
        # Initialize reports
        self.initialize_reports()
        
        # Generate all pipeline combinations
        all_pipelines = self.generate_pipeline_combinations()
        
        print(f"Starting One-vs-Rest classification with {len(all_pipelines)} pipeline combinations...")
        print(f"Number of classes: {n_classes}")
        print(f"OvR re-optimization: {'Enabled' if self.reoptimize_ovr else 'Disabled'}")
        
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
                    
                    # Create base result for main model
                    result = self.create_base_result(
                        scorer_key, estimator, scaler, feature_selector, searcher, scorer, n_classes, position
                    )
                    
                    # Store main model
                    self.main_models[result['key']] = candidate['best_estimator']
                    
                    # Evaluate the main model using the base class method
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
                    
                    # Write main model result to CSV
                    self.write_result_to_csv(result)
                    
                    # Generate OvR models and results
                    print(f"\t\tGenerating OvR models for {n_classes} classes...")
                    
                    csv_entries, class_data, new_ovr_models, additional_fits = self._generate_ovr_models_and_results(
                        pipeline, model['features'], candidate['best_estimator'], result,
                        x_train, y_train, x_val, y_val, x_test, y_test, labels,
                        estimator, scorer, self.reoptimize_ovr
                    )
                    
                    # Update total fits with OvR model fits
                    self.total_fits[estimator] += additional_fits
                    
                    # Store any new OvR models
                    self.ovr_models.update(new_ovr_models)
                    
                    # Write all OvR CSV entries
                    for csv_entry in csv_entries:
                        self.write_result_to_csv(csv_entry)
                    
                    # Save class-specific data as .pkl.gz file
                    self.save_class_results(class_data, result['key'])
                    
                    print(f"\t\tGenerated {len(csv_entries)} OvR models")
        
        # Update metadata with OvR-specific information
        self.parameters['ovr_reoptimized'] = self.reoptimize_ovr
        
        # Finalize reports and save results
        self.finalize_reports(start, n_classes)
        
        print(f"One-vs-Rest classification completed successfully!")
        print(f"Generated {len(self.main_models)} main models and {len(self.ovr_models)} OvR models")
        print(f"Total models: {len(self.main_models) + len(self.ovr_models)}")
        
        return True
