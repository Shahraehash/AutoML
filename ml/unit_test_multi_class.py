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
    'sample-data/multiclass_train.csv', 'sample-data/multiclass_test.csv', LABEL_COLUMN)

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
    assert generalization['accuracy'] == 0.936
    assert generalization['avg_sn_sp'] == 0.9832
    assert generalization['f1'] == 0.7187
    assert generalization['sensitivity'] == 0.7082
    assert generalization['specificity'] == 0.9425

def test_logistic_regression_with_standard_scaler():
    """Test LR with Standard Scaler"""

    generalization = run_pipeline('std', 'none', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.936
    assert generalization['avg_sn_sp'] == 0.9834
    assert generalization['f1'] == 0.7187
    assert generalization['sensitivity'] == 0.7082
    assert generalization['specificity'] == 0.9425

def test_logistic_regression_with_standard_scaler_with_select_75():
    """Test LR with standard scaler and select percentile 75"""

    generalization = run_pipeline('std', 'select-75', 'lr', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9461
    assert generalization['avg_sn_sp'] == 0.9844
    assert generalization['f1'] == 0.7544
    assert generalization['sensitivity'] == 0.7574
    assert generalization['specificity'] == 0.9669

def test_k_nearest_neighbor():
    """Test KNN"""

    generalization = run_pipeline('none', 'none', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9562
    assert generalization['avg_sn_sp'] == 0.9705
    assert generalization['f1'] == 0.7955
    assert generalization['sensitivity'] == 0.8074
    assert generalization['specificity'] == 0.9774

def test_k_nearest_neighbor_with_standard_scaler():
    """Test KNN with standard scaler"""

    generalization = run_pipeline('std', 'none', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9495
    assert generalization['avg_sn_sp'] == 0.9772
    assert generalization['f1'] == 0.7495
    assert generalization['sensitivity'] == 0.7757
    assert generalization['specificity'] == 0.9819

def test_k_nearest_neighbor_with_standard_scaler_with_select_75():
    """Test KNN with standard scaler and select percentile 75%"""

    generalization = run_pipeline('std', 'select-75', 'knn', 'accuracy', 'grid', False)
    assert generalization['accuracy'] == 0.9529
    assert generalization['avg_sn_sp'] == 0.9731
    assert generalization['f1'] == 0.7906
    assert generalization['sensitivity'] == 0.7892
    assert generalization['specificity'] == 0.9693


