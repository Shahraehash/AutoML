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
    assert model['generalization']['f1'] == 0.40740740740740744
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.2

def test_logistic_regression_with_standard_scaler():
    pipeline = generatePipeline('std', 'none', 'lr')
    model = generateModel('lr', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.49019607843137253
    assert model['generalization']['auc'] == 0.675
    assert model['generalization']['f1'] == 0.4583333333333333
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.35