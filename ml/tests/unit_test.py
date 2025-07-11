"""
Unit Tests
"""

from .import_data import import_data
from .generalization import generalize
from .model import generate_model
from .pipeline import generate_pipeline
from .refit import refit_model

# Load the test data
LABEL_COLUMN = 'Cancer'

# Import data
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X2, Y2, FEATURE_NAMES, _ = import_data(
    'sample-data/train.csv', 'sample-data/test.csv', LABEL_COLUMN)

def run_pipeline(scaler, feature_selector, estimator, scoring, searcher, shuffle):
    """Helper method to run unit tests"""

    pipeline = generate_pipeline(scaler, feature_selector, estimator, Y_TRAIN, [scoring], searcher, shuffle)
    model = generate_model(pipeline[0], FEATURE_NAMES, X_TRAIN, Y_TRAIN)
    model.update(refit_model(pipeline[0], model['features'], estimator, scoring, X_TRAIN, Y_TRAIN)[0])
    generalization = generalize(pipeline[0], model['features'], model['best_estimator'], X2, Y2)
    return generalization

# =============================================================================
# LOGISTIC REGRESSION TESTS
# =============================================================================

def test_logistic_regression():
    """Test LR"""

    generalization = run_pipeline('none', 'none', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9933
    assert generalization['avg_sn_sp'] == 0.9961
    assert generalization['f1'] == 0.9861
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9922

def test_logistic_regression_with_standard_scaler():
    """Test LR with Standard Scaler"""

    generalization = run_pipeline('std', 'none', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9933
    assert generalization['avg_sn_sp'] == 0.9961
    assert generalization['f1'] == 0.9861
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9922

def test_logistic_regression_with_standard_scaler_with_select_75():
    """Test LR with standard scaler and select percentile 75"""

    generalization = run_pipeline('std', 'select-75', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9966
    assert generalization['avg_sn_sp'] == 0.998
    assert generalization['f1'] == 0.9930
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9961

# =============================================================================
# K-NEAREST NEIGHBORS TESTS 
# =============================================================================

def test_k_nearest_neighbor():
    """Test KNN"""

    generalization = run_pipeline('none', 'none', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9899
    assert generalization['avg_sn_sp'] == 0.9941
    assert generalization['f1'] == 0.9794
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9883

def test_k_nearest_neighbor_with_standard_scaler():
    """Test KNN with standard scaler"""

    generalization = run_pipeline('std', 'none', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9933
    assert generalization['avg_sn_sp'] == 0.9961
    assert generalization['f1'] == 0.9861
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9922

def test_k_nearest_neighbor_with_standard_scaler_with_select_75():
    """Test KNN with standard scaler and select percentile 75%"""

    generalization = run_pipeline('std', 'select-75', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9865
    assert generalization['avg_sn_sp'] == 0.9922
    assert generalization['f1'] == 0.9728
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9844

# =============================================================================
# SUPPORT VECTOR MACHINE TESTS 
# =============================================================================

def test_support_vector_machine():
    """Test SVM"""

    generalization = run_pipeline('none', 'none', 'svm', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9865
    assert generalization['avg_sn_sp'] == 0.9922
    assert generalization['f1'] == 0.9728
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9844

def test_support_vector_machine_with_standard_scaler():
    """Test SVM with standard scaler"""

    generalization = run_pipeline('std', 'none', 'svm', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9899
    assert generalization['avg_sn_sp'] == 0.9941
    assert generalization['f1'] == 0.9794
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9883

def test_support_vector_machine_with_standard_scaler_and_roc_auc():
    """Test SVM with standard scaler and ROC AUC scoring"""

    generalization = run_pipeline('std', 'none', 'svm', 'roc_auc', 'grid', False)
    assert generalization['accuracy'] == 0.9865
    assert generalization['avg_sn_sp'] == 0.9922
    assert generalization['f1'] == 0.9728
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9844

def test_support_vector_machine_with_standard_scaler_with_select_75():
    """Test SVM with standard scaler and select percentile 75%"""

    generalization = run_pipeline('std', 'select-75', 'svm', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9832
    assert generalization['avg_sn_sp'] == 0.9902
    assert generalization['f1'] == 0.9663
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.9805
