from import_data import importData
from model import generateModel
from pipeline import generatePipeline

# Load the test data
labelColumn = 'AKI'

# Import data
data, data_test, X, Y, X2, Y2, X_train, X_test, Y_train, Y_test = importData('data/train.csv', 'data/test.csv', labelColumn)

def test_logistic_regression():
    pipeline = generatePipeline('none', 'none', 'lr')
    model = generateModel('lr', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.37254901960784315
    assert model['generalization']['auc'] == 0.6
    assert model['generalization']['f1'] == 0.3703703703703704
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.2

def test_logistic_regression_with_standard_scaler():
    pipeline = generatePipeline('std', 'none', 'lr')
    model = generateModel('lr', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.49019607843137253
    assert model['generalization']['auc'] == 0.675
    assert model['generalization']['f1'] == 0.48842592592592593
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.35

def test_k_nearest_neighbor():
    pipeline = generatePipeline('none', 'none', 'knn')
    model = generateModel('knn', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.39215686274509803
    assert model['generalization']['auc'] == 0.6125
    assert model['generalization']['f1'] == 0.39122063919907585
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.225

def test_k_nearest_neighbor_with_standard_scaler():
    pipeline = generatePipeline('std', 'none', 'knn')
    model = generateModel('knn', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.5490196078431373
    assert model['generalization']['auc'] == 0.6795454545454545
    assert model['generalization']['f1'] == 0.5376428852975955
    assert model['generalization']['sensitivity'] == 0.9090909090909091
    assert model['generalization']['specificity'] == 0.45

def test_support_vector_machine():
    pipeline = generatePipeline('none', 'none', 'svm')
    model = generateModel('svm', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.2549019607843137
    assert model['generalization']['auc'] == 0.4920454545454545
    assert model['generalization']['f1'] == 0.24059561128526644
    assert model['generalization']['sensitivity'] == 0.9090909090909091
    assert model['generalization']['specificity'] == 0.075

def test_support_vector_machine_with_standard_scaler():
    pipeline = generatePipeline('std', 'none', 'svm')
    model = generateModel('svm', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.4117647058823529
    assert model['generalization']['auc'] == 0.625
    assert model['generalization']['f1'] == 0.4115384615384615 
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.25