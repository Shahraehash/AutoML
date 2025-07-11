"""
Utility functions for managing class-specific results storage and retrieval
"""

import os
import pickle
import numpy as np
import gzip


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
