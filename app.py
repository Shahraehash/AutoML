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
import itertools

from estimators import estimatorNames
from feature_selection import featureSelectorNames
from model import generateModel
from import_data import importData
from pipeline import generatePipeline
from scalers import scalerNames

#%%
# Define the labels for our classes
# This is used for the classification reproting (more readable then 0/1)
labels = ['No AKI', 'AKI']
labelColumn = 'AKI'

#%%
# Import data
data, data_test, X, Y, X2, Y2, X_train, X_test, Y_train, Y_test = importData('data/train.csv', 'data/test.csv', labelColumn)

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
