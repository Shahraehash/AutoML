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

from sklearn.model_selection import train_test_split

from estimators import estimatorNames
from model import generateModel

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
print('Features used:', ', '.join(sorted(X)))

for algorithm in estimatorNames:
    models[algorithm] = generateModel(algorithm, X_train, Y_train, X, Y, X2, Y2, labels)

#%%
