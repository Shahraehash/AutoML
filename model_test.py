from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from model import generateModel
from pipeline import generatePipeline

# Load the test data
breast_cancer = load_breast_cancer(return_X_y=True)
X = breast_cancer[0]
Y = breast_cancer[1]

X2 = X
Y2 = Y

#%%
# Generate test/train split from the train data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=5, stratify=Y)

# Verify an LR model with no scaler and no feature selector
def test_lr():
    pipeline = generatePipeline('none', 'none', 'lr')
    model = generateModel('lr', pipeline, X_train, Y_train, X, Y, X2, Y2)
    print('hello!!!')
    assert model['generalization']['accuracy'] == ''
    assert model['generalization']['auc'] == ''
    assert model['generalization']['f1'] == ''
    assert model['generalization']['sensitivity'] == ''
    assert model['generalization']['specificity'] == ''