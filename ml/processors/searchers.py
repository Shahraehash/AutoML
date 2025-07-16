"""
All hyper-parameter search methods
"""

import os
import pandas as pd
import tempfile
import multiprocessing
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold

from .estimators import ESTIMATORS, get_xgb_classifier
from .hyperparameters import HYPER_PARAMETER_RANGE

# Define the max iterations for random
MAX_RANDOM_ITERATIONS = 100

# Define the number of splits for the cross validator
N_SPLITS = 10

def get_safe_n_jobs():
    """
    Get safe number of parallel jobs for the current environment
    Handles Celery workers and concurrent job scenarios
    """
    # Check if running in Celery worker
    celery_indicators = [
        'CELERY_WORKER_NAME' in os.environ,
        'celery' in os.environ.get('_', '').lower(),
        'worker' in os.environ.get('CELERY_WORKER_NAME', '').lower(),
        any('celery' in str(arg).lower() for arg in os.sys.argv if hasattr(os, 'sys'))
    ]
    
    if any(celery_indicators):
        # In Celery worker, use limited parallelism to prevent resource conflicts
        cpu_count = multiprocessing.cpu_count()
        # Use half the cores, minimum 1, maximum 4 to prevent resource exhaustion
        safe_jobs = max(1, min(4, cpu_count // 2))
        print(f"Celery worker detected: Using {safe_jobs} parallel jobs (CPU count: {cpu_count})")
        return safe_jobs
    else:
        # Standalone mode, use all cores
        return -1

def configure_joblib_for_worker():
    """
    Configure joblib for safe operation in worker environments
    """
    try:
        # Create worker-specific temporary directory
        worker_id = os.environ.get('CELERY_WORKER_NAME', f'worker_{os.getpid()}')
        temp_dir = os.path.join(tempfile.gettempdir(), f'joblib_{worker_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Configure joblib environment variables for better resource management
        os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
        os.environ['LOKY_MAX_CPU_COUNT'] = str(get_safe_n_jobs() if get_safe_n_jobs() > 0 else 4)
        
        # Configure joblib to be more conservative with resources
        os.environ['JOBLIB_MULTIPROCESSING'] = '1'
        os.environ['LOKY_PICKLER'] = 'pickle'
        
        print(f"Configured joblib temp directory: {temp_dir}")
        return temp_dir
    except Exception as e:
        print(f"Warning: Could not configure joblib worker environment: {e}")
        return None

# Configure joblib when module is imported
configure_joblib_for_worker()

def make_grid_search(estimator, scoring, shuffle, custom_hyper_parameters, y_train):
    """Generate grid search with 10 fold cross validator"""

    # Define the cross validator (shuffle the data between each fold)
    # This reduces correlation between outcome and train data order.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=shuffle)

    if custom_hyper_parameters is not None and\
        'grid' in custom_hyper_parameters and\
        estimator in custom_hyper_parameters['grid']:
        parameter_range = custom_hyper_parameters['grid'][estimator]
    else:
        parameter_range = HYPER_PARAMETER_RANGE['grid'][estimator]\
            if estimator in HYPER_PARAMETER_RANGE['grid'] else {}

    # Handle XGBoost multiclass configuration
    base_estimator = ESTIMATORS[estimator]
    if estimator == 'gb' and y_train is not None:
        n_classes = len(pd.Series(y_train).unique())
        base_estimator = get_xgb_classifier(n_classes)
        
        # Update parameter range for multiclass
        if n_classes > 2 and 'objective' in parameter_range:
            parameter_range = parameter_range.copy()
            parameter_range['objective'] = ['multi:softprob']

    # Use safe number of jobs to prevent resource conflicts
    n_jobs = get_safe_n_jobs()
    
    return (
        GridSearchCV(
            base_estimator,
            parameter_range,
            cv=cv,
            scoring=scoring,
            refit=False,
            n_jobs=n_jobs,
            return_train_score=False
        ),
        len(list(ParameterGrid(parameter_range))) *\
            cv.get_n_splits()
    )

def make_random_search(estimator, scoring, shuffle, custom_hyper_parameters, y_train):
    """Generate random search with defined max iterations"""

    # Define the cross validator (shuffle the data between each fold)
    # This reduces correlation between outcome and train data order.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=shuffle)

    if custom_hyper_parameters is not None and\
        'random' in custom_hyper_parameters and\
        estimator in custom_hyper_parameters['random']:
        parameter_range = custom_hyper_parameters['random'][estimator]
    else:
        parameter_range = HYPER_PARAMETER_RANGE['random'][estimator]\
            if estimator in HYPER_PARAMETER_RANGE['random'] else {}

    if callable(parameter_range):
        parameter_range = parameter_range(pd.Series(y_train).value_counts().min())

    # Handle XGBoost multiclass configuration
    base_estimator = ESTIMATORS[estimator]
    if estimator == 'gb' and y_train is not None:
        n_classes = len(pd.Series(y_train).unique())
        base_estimator = get_xgb_classifier(n_classes)
        
        # Update parameter range for multiclass
        if n_classes > 2 and 'objective' in parameter_range:
            parameter_range = parameter_range.copy()
            parameter_range['objective'] = ['multi:softprob']

    # When the grid contains an RVS method, the parameter grid cannot generate
    # an exhaustive list and throws an error. In this case, iterate the max
    # count allowed.
    try:
        total_range = len(list(ParameterGrid(parameter_range)))
        iterations = total_range if MAX_RANDOM_ITERATIONS >= total_range else MAX_RANDOM_ITERATIONS
    except Exception:
        iterations = MAX_RANDOM_ITERATIONS

    # Use safe number of jobs to prevent resource conflicts
    n_jobs = get_safe_n_jobs()
    
    return (
        RandomizedSearchCV(
            base_estimator,
            parameter_range,
            cv=cv,
            scoring=scoring,
            refit=False,
            n_iter=iterations,
            n_jobs=n_jobs,
            return_train_score=False
        ),
        iterations * cv.get_n_splits()
    )

SEARCHERS = {
    'grid': make_grid_search,
    'random': make_random_search,
    'random2': make_random_search
}

SEARCHER_NAMES = {
    'grid': 'grid search',
    'random': 'random search',
    'random2': '2nd random search'
}
