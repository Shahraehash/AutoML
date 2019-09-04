#%%
# Auto ML
#
# Supervised learning using an exhaustive search of ideal pre-processing (if any), algorithms,
# and hyper-parameters with feature engineering.

#%%
# Hide warning from the output
import warnings
warnings.filterwarnings('ignore')

#%%
# Dependencies
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split

from estimators import estimatorNames
from feature_selection import featureSelectorNames
from model import generateModel
from pipeline import generatePipeline
from scalers import scalerNames

#%%
# Define the labels for our classes
# This is used for the classification reproting (more readable then 0/1)
labels = ['No AKI', 'AKI']
labelColumn = 'AKI'

#%%
# Import data
data = pd.read_csv('data/train.csv').dropna()
X = data.drop(labelColumn, axis=1)
Y = data[labelColumn]

data_test = pd.read_csv('data/test.csv').dropna()
X2 = data_test.drop(labelColumn, axis=1)
Y2 = data_test[labelColumn]

#%%
# Generate test/train split from the train data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=5, stratify=Y)

#%%
# Generate all models
models = {}
for scaler, featureSelector, estimator in list(itertools.product(*[scalerNames, featureSelectorNames, estimatorNames])):
    print('Generating ' + estimatorNames[estimator] + ' model with ' + scalerNames[scaler] + ' and with ' + featureSelectorNames[featureSelector])

    if not scaler in models:
        models[scaler] = {}
    
    if not featureSelector in models[scaler]:
        models[scaler][featureSelector] = {}

    pipeline = generatePipeline(scaler, featureSelector, estimator)
    models[scaler][featureSelector][estimator] = generateModel(estimator, pipeline, X_train, Y_train, X, Y, X2, Y2, labels)
