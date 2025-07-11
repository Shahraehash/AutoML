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
    generalization = generalize(pipeline[0], model['features'], model['best_estimator'], X2, Y2)
    return generalization

THRESHOLD = 0.05

# =============================================================================
# LOGISTIC REGRESSION TESTS
# =============================================================================

def test_logistic_regression():
    """Test Logistic Regression"""
    generalization = run_pipeline('none', 'none', 'lr', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.936) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9832) < THRESHOLD
    assert abs(generalization['f1'] - 0.7187) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7082) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9425) < THRESHOLD

def test_logistic_regression_with_standard_scaler():
    """Test Logistic Regression with Standard Scaler"""
    generalization = run_pipeline('std', 'none', 'lr', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.936) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9834) < THRESHOLD
    assert abs(generalization['f1'] - 0.7187) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7082) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9425) < THRESHOLD

def test_logistic_regression_with_minmax_scaler():
    """Test Logistic Regression with MinMax Scaler"""
    generalization = run_pipeline('minmax', 'none', 'lr', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.936) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9827) < THRESHOLD
    assert abs(generalization['f1'] - 0.7187) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7082) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9425) < THRESHOLD

def test_logistic_regression_with_feature_selection():
    """Test Logistic Regression with feature selection"""
    generalization = run_pipeline('std', 'select-75', 'lr', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9461) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9844) < THRESHOLD
    assert abs(generalization['f1'] - 0.7544) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7574) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9669) < THRESHOLD

def test_logistic_regression_with_pca():
    """Test Logistic Regression with PCA"""
    generalization = run_pipeline('std', 'pca-80', 'lr', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9394) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9859) < THRESHOLD
    assert abs(generalization['f1'] - 0.7012) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7265) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9783) < THRESHOLD


# =============================================================================
# K-NEAREST NEIGHBORS TESTS 
# =============================================================================

def test_k_nearest_neighbor():
    """Test K-Nearest Neighbors"""
    generalization = run_pipeline('none', 'none', 'knn', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9562) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9705) < THRESHOLD
    assert abs(generalization['f1'] - 0.7955) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8074) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9774) < THRESHOLD

def test_k_nearest_neighbor_with_standard_scaler():
    """Test KNN with Standard Scaler"""
    generalization = run_pipeline('std', 'none', 'knn', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9495) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9772) < THRESHOLD
    assert abs(generalization['f1'] - 0.7495) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7757) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9819) < THRESHOLD
    
def test_k_nearest_neighbor_with_minmax_scaler():
    """Test KNN with MinMax Scaler"""
    generalization = run_pipeline('minmax', 'none', 'knn', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9562) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9807) < THRESHOLD
    assert abs(generalization['f1'] - 0.7894) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8074) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9844) < THRESHOLD

def test_k_nearest_neighbor_with_feature_selection():
    """Test KNN with feature selection"""
    generalization = run_pipeline('std', 'select-75', 'knn', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9529) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9731) < THRESHOLD
    assert abs(generalization['f1'] - 0.7906) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7892) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9693) < THRESHOLD

# =============================================================================
# SUPPORT VECTOR MACHINE TESTS 
# =============================================================================

def test_support_vector_machine():
    """Test Support Vector Machine"""
    generalization = run_pipeline('none', 'none', 'svm', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9428) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9882) < THRESHOLD
    assert abs(generalization['f1'] - 0.7637) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7392) < THRESHOLD
    assert abs(generalization['specificity'] - 0.938) < THRESHOLD

def test_support_vector_machine_with_standard_scaler():
    """Test SVM with Standard Scaler"""
    generalization = run_pipeline('std', 'none', 'svm', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9428) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9877) < THRESHOLD
    assert abs(generalization['f1'] - 0.7637) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7392) < THRESHOLD
    assert abs(generalization['specificity'] - 0.938) < THRESHOLD

def test_support_vector_machine_with_minmax_scaler():
    """Test SVM with MinMax Scaler"""
    generalization = run_pipeline('minmax', 'none', 'svm', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9495) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9885) < THRESHOLD
    assert abs(generalization['f1'] - 0.792) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7725) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9473) < THRESHOLD

def test_support_vector_machine_with_feature_selection():
    """Test SVM with feature selection"""
    generalization = run_pipeline('std', 'select-75', 'svm', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9596) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9898) < THRESHOLD
    assert abs(generalization['f1'] - 0.8228) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8032) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9578) < THRESHOLD

def test_support_vector_machine_with_roc_auc():
    """Test SVM with ROC AUC scoring"""
    generalization = run_pipeline('std', 'none', 'svm', 'roc_auc', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9495) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.988) < THRESHOLD
    assert abs(generalization['f1'] - 0.7752) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7741) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9612) < THRESHOLD

# =============================================================================
# GRADIENT BOOSTING TESTS
# =============================================================================

def test_gradient_boosting():
    """Test Gradient Boosting Machine"""
    generalization = run_pipeline('none', 'none', 'gb', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9697) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9939) < THRESHOLD
    assert abs(generalization['f1'] - 0.8872) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8693) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9615) < THRESHOLD

def test_gradient_boosting_with_minmax_scaler():
    """Test Gradient Boosting with MinMax Scaler"""
    generalization = run_pipeline('minmax', 'none', 'gb', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9697) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9939) < THRESHOLD
    assert abs(generalization['f1'] - 0.8872) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8693) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9615) < THRESHOLD

def test_gradient_boosting_with_roc_auc():
    """Test Gradient Boosting with ROC AUC scoring"""
    generalization = run_pipeline('none', 'none', 'gb', 'roc_auc', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9697) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9939) < THRESHOLD
    assert abs(generalization['f1'] - 0.8872) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8693) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9615) < THRESHOLD

# =============================================================================
# NAIVE BAYES TESTS
# =============================================================================

def test_naive_bayes():
    """Test Naive Bayes"""
    generalization = run_pipeline('none', 'none', 'nb', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.936) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9863) < THRESHOLD
    assert abs(generalization['f1'] - 0.7357) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7697) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9771) < THRESHOLD

def test_naive_bayes_with_minmax_scaler():
    """Test Naive Bayes with MinMax Scaler"""
    generalization = run_pipeline('minmax', 'none', 'nb', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.936) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9863) < THRESHOLD
    assert abs(generalization['f1'] - 0.7357) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7697) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9771) < THRESHOLD

def test_naive_bayes_with_feature_selection():
    """Test Naive Bayes with feature selection"""
    generalization = run_pipeline('std', 'select-75', 'nb', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9529) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.989) < THRESHOLD
    assert abs(generalization['f1'] - 0.792) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8037) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9831) < THRESHOLD

def test_naive_bayes_with_roc_auc():
    """Test Naive Bayes with ROC AUC scoring"""
    generalization = run_pipeline('std', 'none', 'nb', 'roc_auc', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9495) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.988) < THRESHOLD
    assert abs(generalization['f1'] - 0.7752) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7741) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9612) < THRESHOLD

# =============================================================================
# NEURAL NETWORK TESTS
# =============================================================================

def test_neural_network():
    """Test Neural Network (MLP)"""
    generalization = run_pipeline('none', 'none', 'mlp', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9327) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9769) < THRESHOLD
    assert abs(generalization['f1'] - 0.707) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.6923) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9274) < THRESHOLD

def test_neural_network_with_standard_scaler():
    """Test Neural Network with Standard Scaler"""
    generalization = run_pipeline('std', 'none', 'mlp', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9428) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9877) < THRESHOLD
    assert abs(generalization['f1'] - 0.7637) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7392) < THRESHOLD
    assert abs(generalization['specificity'] - 0.938) < THRESHOLD

def test_neural_network_with_feature_selection():
    """Test Neural Network with feature selection"""
    generalization = run_pipeline('std', 'select-75', 'mlp', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9495) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9845) < THRESHOLD
    assert abs(generalization['f1'] - 0.7628) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7733) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9819) < THRESHOLD

def test_neural_network_with_pca():
    """Test Neural Network with PCA"""
    generalization = run_pipeline('std', 'pca-80', 'mlp', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9495) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9832) < THRESHOLD
    assert abs(generalization['f1'] - 0.7555) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.7749) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9819) < THRESHOLD

# =============================================================================
# RANDOM FOREST TESTS 
# =============================================================================

def test_random_forest():
    """Test Random Forest"""
    generalization = run_pipeline('none', 'none', 'rf', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9697) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.995) < THRESHOLD
    assert abs(generalization['f1'] - 0.8691) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8693) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9822) < THRESHOLD

def test_random_forest_with_standard_scaler():
    """Test Random Forest with Standard Scaler"""
    generalization = run_pipeline('std', 'none', 'rf', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9697) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9944) < THRESHOLD
    assert abs(generalization['f1'] - 0.8691) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8693) < THRESHOLD
    assert abs(generalization['specificity'] - 0.9822) < THRESHOLD

def test_random_forest_with_feature_selection():
    """Test Random Forest with feature selection"""
    generalization = run_pipeline('std', 'select-75', 'rf', 'accuracy', 'grid', False)
    assert abs(generalization['accuracy'] - 0.9663) < THRESHOLD
    assert abs(generalization['avg_sn_sp'] - 0.9933) < THRESHOLD
    assert abs(generalization['f1'] - 0.8527) < THRESHOLD
    assert abs(generalization['sensitivity'] - 0.8527) < THRESHOLD
    assert abs(generalization['specificity'] - 0.981) < THRESHOLD
