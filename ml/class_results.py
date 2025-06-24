"""
Utility functions for managing class-specific results storage and retrieval
"""

import os
import pickle
import gzip
from .reliability import reliability
from .precision import precision_recall
from .roc import roc


def compute_class_specific_results(pipeline, features, estimator, x_test, y_test, n_classes, model_key):
    """Compute class-specific OvR results for all classes"""
    
    class_results = {
        'model_key': model_key,
        'n_classes': n_classes,
        'class_data': {}
    }
    
    for class_idx in range(n_classes):
        try:
            # Compute reliability, precision_recall, and roc for this class
            reliability_data = reliability(pipeline, features, estimator, x_test, y_test, class_idx)
            precision_data = precision_recall(pipeline, features, estimator, x_test, y_test, class_idx)
            roc_data = roc(pipeline, features, estimator, x_test, y_test, class_idx)
            
            class_results['class_data'][class_idx] = {
                'reliability': reliability_data,
                'precision_recall': precision_data,
                'roc_auc': roc_data
            }
        except Exception as e:
            print(f"Error computing class-specific results for class {class_idx} in model {model_key}: {e}")
            # Store empty results for this class to maintain consistency
            class_results['class_data'][class_idx] = {
                'reliability': {'brier_score': 0, 'fop': [], 'mpv': []},
                'precision_recall': {'precision': [], 'recall': [], 'thresholds': [], 'auc': 0},
                'roc_auc': {'fpr': [], 'tpr': [], 'roc_auc': 0}
            }
    
    return class_results


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


def cleanup_class_results(class_results_dir):
    """Remove all class results for a job"""
    
    if os.path.exists(class_results_dir):
        try:
            import shutil
            shutil.rmtree(class_results_dir)
        except Exception as e:
            print(f"Error cleaning up class results: {e}")
