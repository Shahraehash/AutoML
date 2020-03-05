"""
Unit Tests
"""

from .import_data import import_data
from .generalization import generalize
from .model import generate_model
from .pipeline import generate_pipeline
from .refit import refit_model

# Load the test data
LABEL_COLUMN = 'AKI'

# Import data
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X2, Y2, FEATURE_NAMES, _ = import_data(
    'sample-data/train.csv', 'sample-data/test.csv', LABEL_COLUMN)

def run_pipeline(scaler, feature_selector, estimator, scoring, searcher, shuffle):
    """Helper method to run unit tests"""

    pipeline = generate_pipeline(scaler, feature_selector, estimator, Y_TRAIN, [scoring], searcher, shuffle)
    model = generate_model(pipeline[0], FEATURE_NAMES, X_TRAIN, Y_TRAIN)
    model.update(refit_model(pipeline[0], model['features'], estimator, scoring, X_TRAIN, Y_TRAIN)[0])
    generalization = generalize(model['features'], model['best_estimator'], pipeline[0], X2, Y2)
    return generalization

def test_logistic_regression():
    """Test LR"""

    generalization = run_pipeline('none', 'none', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.3725
    assert generalization['avg_sn_sp'] == 0.6
    assert generalization['f1'] == 0.3704
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.2

def test_logistic_regression_with_standard_scaler():
    """Test LR with Standard Scaler"""

    generalization = run_pipeline('std', 'none', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.4902
    assert generalization['avg_sn_sp'] == 0.675
    assert generalization['f1'] == 0.4884
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.35

def test_logistic_regression_with_standard_scaler_with_select_75():
    """Test LR with standard scaler and select percentile 75"""

    generalization = run_pipeline('std', 'select-75', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.3529
    assert generalization['avg_sn_sp'] == 0.5875
    assert generalization['f1'] == 0.3489
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.175

def test_k_nearest_neighbor():
    """Test KNN"""

    generalization = run_pipeline('none', 'none', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.3922
    assert generalization['avg_sn_sp'] == 0.6125
    assert generalization['f1'] == 0.3912
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.225

def test_k_nearest_neighbor_with_standard_scaler():
    """Test KNN with standard scaler"""

    generalization = run_pipeline('std', 'none', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.5490
    assert generalization['avg_sn_sp'] == 0.6795
    assert generalization['f1'] == 0.5376
    assert generalization['sensitivity'] == 0.9091
    assert generalization['specificity'] == 0.45

def test_k_nearest_neighbor_with_standard_scaler_with_select_75():
    """Test KNN with standard scaler and select percentile 75%"""

    generalization = run_pipeline('std', 'select-75', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.4902
    assert generalization['avg_sn_sp'] == 0.675
    assert generalization['f1'] == 0.4884
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.35

def test_support_vector_machine():
    """Test SVM"""

    generalization = run_pipeline('none', 'none', 'svm', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.2549
    assert generalization['avg_sn_sp'] == 0.4920
    assert generalization['f1'] == 0.2406
    assert generalization['sensitivity'] == 0.9091
    assert generalization['specificity'] == 0.075

def test_support_vector_machine_with_standard_scaler():
    """Test SVM with standard scaler"""

    generalization = run_pipeline('std', 'none', 'svm', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.4118
    assert generalization['avg_sn_sp'] == 0.625
    assert generalization['f1'] == 0.4115
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.25

def test_support_vector_machine_with_standard_scaler_and_roc_auc():
    """Test SVM with standard scaler and ROC AUC scoring"""

    generalization = run_pipeline('std', 'none', 'svm', 'roc_auc', 'grid', False)
    assert generalization['accuracy'] == 0.7059
    assert generalization['avg_sn_sp'] == 0.6807
    assert generalization['f1'] == 0.6386
    assert generalization['sensitivity'] == 0.6364
    assert generalization['specificity'] == 0.725

def test_support_vector_machine_with_standard_scaler_with_select_75():
    """Test SVM with standard scaler and select percentile 75%"""

    generalization = run_pipeline('std', 'select-75', 'svm', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.2353
    assert generalization['avg_sn_sp'] == 0.5125
    assert generalization['f1'] == 0.2047
    assert generalization['sensitivity'] == 1.0
    assert generalization['specificity'] == 0.025
