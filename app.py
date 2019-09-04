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
from os import path
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectPercentile

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from estimators import estimators, estimatorNames
from hyperparameters import hyperParameterRange

#%%
# Define the cross validator
cv = StratifiedKFold(n_splits=10)

#%%
# Define the labels for our classes
# This is used for the classification reproting (more readable then 0/1)
labels = ['No AKI', 'AKI']

#%%
# Define the major data containers needed
# This will house the various modifications
# of each data item
dataTypes = ['raw', 'scaled', 'selected-scaled']
data = {}
data_test = {}
X = {}
Y = {}
X2 = {}
Y2 = {}
X_train = {}
X_test = {}
Y_train = {}
Y_test = {}

#%%
# Generate the X and Y variables by splitting the
# features from the labels
def generateXY(X, Y, data, type, labelColumn):
    X[type] = data[type].drop(labelColumn, axis=1)
    Y[type] = data[type][labelColumn]

# Generate a test/train split for the keyed data
def generateTestSplit(type):
    X_train[type], X_test[type], Y_train[type], Y_test[type] = train_test_split(X[type], Y[type], test_size=.2, random_state=5, stratify=Y[type])

# Generate the scaled data for the provided data
# set and store and the provided output
def generateScaledData(sourceType, destinationType):
    sc = StandardScaler()
    X_train[destinationType] = sc.fit_transform(X_train[sourceType])
    X_test[destinationType] = sc.transform(X_test[sourceType])
    X2[destinationType] = sc.transform(X2[sourceType])

# Small method to allow a fallback value when data is not available
# in the selected type. This happens when there is no difference
# between the adjusted data and raw data. Eg. labels (Y), original data
# set when values are scaled (X), labels from the train/test split (Y_train),
# and labels from the test data set (Y2)
def getValue(dictionary, type):
    fallback = 'selected' if type == 'selected-scaled' else 'raw'        
    return dictionary.get(type, dictionary[fallback])

# Imports a CSV file, drops the null values, and generate the X/Y variables
def importData(type, file, labelColumn):
    sourceData = data if type == 'train' else data_test
    sourceData['raw'] = pd.read_csv(file).dropna()
    generateXY(X if type == 'train' else X2, Y if type == 'train' else Y2, sourceData, 'raw', labelColumn)

#%%
# Import data
importData('train', 'data/train.csv', 'AKI')
importData('test', 'data/test.csv', 'AKI')

#%%
# Generate test/train split from the train data
generateTestSplit('raw')

#%%
# Scale the data (train and test)
generateScaledData('raw', 'scaled')

#%%
# Feature engineering with select percentile
select = SelectPercentile(percentile=75)
select.fit(X_train['raw'], Y_train['raw'])

# Initialize the selected data from the raw data
data['selected'] = data['raw']
data_test['selected'] = data_test['raw']

# Identify the select features/columns and remove them
selectedFeatures = pd.Series(select.get_support(), index=list(X['raw']))
for feature, selected in selectedFeatures.items():
    if not selected:
        data['selected'] = data['selected'].drop(feature, axis=1)
        data_test['selected'] = data_test['selected'].drop(feature, axis=1)

generateXY(X, Y, data, 'selected', 'AKI')
generateXY(X2, Y2, data_test, 'selected', 'AKI')
generateTestSplit('selected')
generateScaledData('selected', 'selected-scaled')

#%%
# Define the generic method to generate the best model for the provided estimator
def generateModel(algorithm, X_train, Y_train, X, Y, X2, Y2):
    print('\tGenerating ' + estimatorNames[algorithm] + ' model:')

    model = estimators[algorithm]()
    model.fit(X_train, Y_train)
    model_cv = cross_val_score(model, X, Y, cv=cv, scoring='accuracy')
    best_params = {}
    performance = {}

    print("\t\tDefault CV Accuracy: %.7g (sd=%.7g)" % (np.mean(model_cv), np.std(model_cv)))

    # Perform a grid search if the algorithm has tunable hyper-parameters
    if algorithm in hyperParameterRange:

        # The parameter `return_train_score` is False because 
        # it's not required and reduces CPU time without it
        model_gs = GridSearchCV(
            model,
            hyperParameterRange[algorithm],
            return_train_score='False',
            cv=cv,
            n_jobs=-1,
            scoring='accuracy'
        )
        model_gs.fit(X_train, Y_train)
        model_gs_cv = cross_val_score(model_gs.best_estimator_, X, Y, cv=cv, scoring='roc_auc')

        performance = pd.DataFrame(model_gs.cv_results_)[['mean_test_score', 'std_test_score']].sort_values(by='mean_test_score', ascending=False)
        model_best = model_gs.best_estimator_
        best_params = model_gs.best_params_

        print("\t\tGridSearchCV AUC: %.7g (sd=%.7g)" % (np.mean(model_gs_cv), np.std(model_gs_cv)))
        print('\t\tBest accuracy: %.7g (sd=%.7g)'
            % (performance.iloc[0]['mean_test_score'], performance.iloc[0]['std_test_score']))
        print('\t\tBest parameters:', json.dumps(best_params, indent=4, sort_keys=True).replace('\n', '\n\t\t'))
    else:
        print('\t\tNo hyper-parameters to tune for this estimator\n')
        model_best = model

    predictions = model_best.predict(X2)
    print('\t\t', classification_report(Y2, predictions, target_names=labels).replace('\n', '\n\t\t'))

    accuracy = accuracy_score(Y2, predictions)
    print('\t\tGeneralization accuracy:', accuracy)

    auc = roc_auc_score(Y2, predictions)
    print('\t\tGeneralization AUC:', auc, '\n')

    return {
        'estimator': model_best,
        'best_params': best_params,
        'performance': performance,
        'generalization': {
            'accuracy': accuracy,
            'auc': auc
        }
    }

#%%
# Generate the best model with unscaled, scaled and selected & scaled data
models = {}
for type in dataTypes:
    print('Generating all models for', type, 'data...')

    print('Features used:', ', '.join(sorted(getValue(X, type))))

    models[type] = {}
    for algorithm in estimatorNames:
        models[type][algorithm] = generateModel(algorithm, getValue(X_train, type), getValue(Y_train, type), getValue(X, type), getValue(Y, type), getValue(X2, type), getValue(Y2, type))


#%%
