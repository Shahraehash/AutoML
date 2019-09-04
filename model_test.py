from model import generateModel
from pipeline import generatePipeline

# Load the test data

# Verify an LR model with no scaler and no feature selector
def test_lr():
    pipeline = generatePipeline('none', 'none', 'lr')
    model = generateModel('lr', pipeline, X_train, Y_train, X, Y, X2, Y2, labels)
    assert model['generalization']['accuracy'] == ''
    assert model['generalization']['auc'] == ''
    assert model['generalization']['f1'] == ''
    assert model['generalization']['sensitivity'] == ''
    assert model['generalization']['specificity'] == ''