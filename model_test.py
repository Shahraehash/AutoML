import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from model import generateModel
from pipeline import generatePipeline

# Load the test data
labelColumn = 'AKI'

data = pd.read_csv('data/train.csv').dropna()
X = data.drop(labelColumn, axis=1)
Y = data[labelColumn]

data_test = pd.read_csv('data/test.csv').dropna()
X2 = data_test.drop(labelColumn, axis=1)
Y2 = data_test[labelColumn]

#%%
# Generate test/train split from the train data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=5, stratify=Y)

# Verify an LR model with no scaler and no feature selector
def test_lr():
    pipeline = generatePipeline('none', 'none', 'lr')
    model = generateModel('lr', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.37254901960784315
    assert model['generalization']['auc'] == 0.6
    assert model['generalization']['f1'] == 0.40740740740740744
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.2

def test_std_lr():
    pipeline = generatePipeline('std', 'none', 'lr')
    model = generateModel('lr', pipeline, X_train, Y_train, X, Y, X2, Y2)
    assert model['generalization']['accuracy'] == 0.49019607843137253
    assert model['generalization']['auc'] == 0.675
    assert model['generalization']['f1'] == 0.4583333333333333
    assert model['generalization']['sensitivity'] == 1.0
    assert model['generalization']['specificity'] == 0.35