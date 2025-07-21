"""
Utility functions for managing class-specific results storage and retrieval
"""

import os
import pickle
import numpy as np
import gzip
from sklearn.pipeline import Pipeline
from .reliability import reliability
from .precision import precision_recall
from .roc import roc
from .refit import refit_model
from .generalization import generalize


def compute_binary_class_results(pipeline, features, estimator, x_val, y_val, x_train=None, y_train=None, x_test=None, y_test=None):
    """Compute metrics for binary classification (used for OvR models in re-optimization mode)"""
    
    # Compute reliability, precision_recall, and roc for binary classification
    generalization_data = generalize(pipeline, features, estimator, x_val, y_val)
    reliability_data = reliability(pipeline, features, estimator, x_val, y_val)
    precision_data = precision_recall(pipeline, features, estimator, x_val, y_val)
    roc_data = roc(pipeline, features, estimator, x_val, y_val)
    
    # Compute training ROC AUC if training data is provided
    training_roc_auc = None
    roc_delta = None
    
    if x_train is not None and y_train is not None:
        training_roc_data = roc(pipeline, features, estimator, x_train, y_train)
        training_roc_auc = training_roc_data['roc_auc']
        
        # Calculate ROC delta
        if roc_data['roc_auc'] is not None and training_roc_auc is not None:
            roc_delta = round(abs(roc_data['roc_auc'] - training_roc_auc), 4)
    
    # Compute test ROC metrics if test data is provided
    test_roc_data = None
    if x_test is not None and y_test is not None:
        test_roc_data = roc(pipeline, features, estimator, x_test, y_test)
    
    return {
        'generalization': generalization_data,
        'reliability': reliability_data,
        'precision_recall': precision_data,
        'roc_auc': roc_data,
        'training_roc_auc': training_roc_auc,
        'roc_delta': roc_delta,
        'test_roc_data': test_roc_data
    }


def compute_class_specific_results(pipeline, features, estimator, x_val, y_val, n_classes, model_key, class_idx, x_train=None, y_train=None, x_test_orig=None, y_test_orig=None):
    generalization_data = generalize(pipeline, features, estimator, x_val, y_val, class_index=class_idx)
    reliability_data = reliability(pipeline, features, estimator, x_val, y_val, class_idx)
    precision_data = precision_recall(pipeline, features, estimator, x_val, y_val, class_idx)
    roc_data = roc(pipeline, features, estimator, x_val, y_val, class_idx)
    
    # Compute training ROC AUC for this class if training data is provided
    training_roc_auc = None
    roc_delta = None
    
    if x_train is not None and y_train is not None:
        training_roc_data = roc(pipeline, features, estimator, x_train, y_train, class_idx)
        training_roc_auc = training_roc_data['roc_auc']
        
        # Calculate ROC delta (absolute difference between generalization and training)
        if roc_data['roc_auc'] is not None and training_roc_auc is not None:
            roc_delta = round(abs(roc_data['roc_auc'] - training_roc_auc), 4)
    
    # Compute test ROC metrics if original test data is provided
    test_roc_data = None
    if x_test_orig is not None and y_test_orig is not None:
        test_roc_data = roc(pipeline, features, estimator, x_test_orig, y_test_orig, class_idx)
    
    return {
        'generalization': generalization_data,
        'reliability': reliability_data,
        'precision_recall': precision_data,
        'roc_auc': roc_data,
        'training_roc_auc': training_roc_auc,
        'roc_delta': roc_delta,
        'test_roc_data': test_roc_data
    }


def save_class_results(class_data, output_dir, model_key):
    """Save class-specific results with compression"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = f"{output_dir}/{model_key}.pkl.gz"
    try:
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(class_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving class results for {model_key}: {e}")


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


def generate_ovr_models_and_results(
    pipeline, features, main_model, main_result, 
    x_train, y_train, x_test, y_test, x2, y2, labels,
    estimator, scorer, reoptimize_ovr=False, custom_labels=None, label_mapping_info=None
):
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
        
        if reoptimize_ovr:
            # Re-optimization mode: Train actual OvR model
            ovr_candidates = refit_model(
                pipeline, features, estimator, scorer, x_train, y_binary
            )
            
            ovr_pipeline_steps = pipeline.steps[:-1] + [('estimator', ovr_candidates[0]['best_estimator'])]
            ovr_model = Pipeline(ovr_pipeline_steps)
            ovr_best_params = ovr_candidates[0]['best_params']
            total_fits += len(ovr_candidates)
            
            # Store OvR model
            ovr_key = f"{main_result['key']}_ovr_class_{class_idx}"
            ovr_models[ovr_key] = ovr_model

            # Re-optimization mode: Use binary classification path
            # OvR model was trained on binary data, so evaluate it on binary data
            class_metrics = compute_binary_class_results(
                pipeline, features, ovr_candidates[0]['best_estimator'], 
                x2, y2_binary,           # Binary generalization data
                x_train, y_binary,       # Binary training data
                x_test, y_test_binary    # Binary test data
            )
            
        else:
            # Efficient mode: Use main model
            ovr_model = main_model
            ovr_best_params = main_result['best_params']
        
            # Efficient mode: Use multiclass classification path with class_idx
            class_metrics = compute_class_specific_results(
                pipeline, features, ovr_model, x2, y2, n_classes, 
                main_result['key'], class_idx, x_train, y_train, x_test, y_test
            )
        
        # Store class data for .pkl.gz file
        all_class_data['class_data'][class_idx] = class_metrics
        
        # Create CSV entry for this OvR model using already computed metrics
        csv_entry = create_ovr_csv_entry(
            main_result, class_idx, ovr_best_params, class_metrics, custom_labels, label_mapping_info
        )
        
        csv_entries.append(csv_entry)
    
    return csv_entries, all_class_data, ovr_models, total_fits


def get_class_label(class_index, custom_labels, label_mapping_info=None):
    """Helper function to get custom class label or fall back to generic label"""
    if class_index is None:
        return None
    
    # If we have custom labels, we need to map from normalized index back to original label
    if custom_labels:
        original_label = class_index  # Default to class_index if no mapping available
        
        # If we have label mapping info, use inverse mapping to get original label
        if label_mapping_info and 'inverse_mapping' in label_mapping_info:
            original_label = label_mapping_info['inverse_mapping'].get(class_index, class_index)
        
        # Look up custom label using original label as key
        original_label_str = str(original_label)
        if original_label_str in custom_labels:
            return custom_labels[original_label_str]
        
        # Also try looking up by normalized index for backward compatibility
        if str(class_index) in custom_labels:
            return custom_labels[str(class_index)]
    
    return f"Class {class_index}"


def create_ovr_csv_entry(main_result, class_idx, ovr_best_params, class_metrics, custom_labels=None, label_mapping_info=None):
    """Create a CSV entry for an OvR model using already computed metrics"""

    # Create base CSV entry
    ovr_result = {
        'key': f"{main_result['key']}_ovr_class_{class_idx}",
        'class_type': 'ovr',
        'class_index': class_idx,
        'class_label': get_class_label(class_idx, custom_labels, label_mapping_info),
        'scaler': main_result['scaler'],
        'feature_selector': main_result['feature_selector'],
        'algorithm': main_result['algorithm'],
        'searcher': main_result['searcher'],
        'scorer': main_result['scorer']
    }
    
    ovr_result.update(class_metrics.get('generalization'))
    
    # Add basic metadata
    ovr_result.update({
        'selected_features': main_result['selected_features'],
        'feature_scores': main_result['feature_scores'],
        'best_params': ovr_best_params
    })
    
    test_data = class_metrics['test_roc_data']
    ovr_result['test_fpr'] = test_data.get('fpr')
    ovr_result['test_tpr'] = test_data.get('tpr')

    ovr_result['training_roc_auc'] = class_metrics.get('training_roc_auc')
    ovr_result['roc_delta'] = class_metrics.get('roc_delta')

    roc_data = class_metrics['roc_auc']
    ovr_result['generalization_fpr'] = roc_data.get('fpr')
    ovr_result['generalization_tpr'] = roc_data.get('tpr')
    
    reliability_data = class_metrics['reliability']
    ovr_result['brier_score'] = reliability_data.get('brier_score')
    ovr_result['fop'] = reliability_data.get('fop')
    ovr_result['mpv'] = reliability_data.get('mpv')

    pr_data = class_metrics['precision_recall']
    ovr_result['precision'] = pr_data.get('precision')
    ovr_result['recall'] = pr_data.get('recall')
    
    
    return ovr_result


def cleanup_class_results(class_results_dir):
    """Remove all class results for a job"""
    
    if os.path.exists(class_results_dir):
        try:
            import shutil
            shutil.rmtree(class_results_dir)
        except Exception as e:
            print(f"Error cleaning up class results: {e}")
