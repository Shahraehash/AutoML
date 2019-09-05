# Dependencies
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def importData(trainPath, testPath, labelColumn):
    data = pd.read_csv(trainPath).dropna()
    X = data.drop(labelColumn, axis=1)
    Y = data[labelColumn]

    data_test = pd.read_csv(testPath).dropna()
    X2 = data_test.drop(labelColumn, axis=1)
    Y2 = data_test[labelColumn]

    # Only keep numeric inputs
    X = X.loc[:, (X.dtypes == np.int64) | (X.dtypes == np.float64)].dropna()
    X2 = X2.loc[:, (X2.dtypes == np.int64) | (X2.dtypes == np.float64)].dropna()

    negativeCount = data[data[labelColumn] == 0].shape[0]
    positiveCount = data[data[labelColumn] == 1].shape[0]

    print('Negative Cases: %.7g\nPositive Cases: %.7g' % (negativeCount, positiveCount))

    if negativeCount / positiveCount < .9:
        print('Warning: Classes are not balanced.')

    # Generate test/train split from the train data
    return [data, data_test, X, Y, X2, Y2] + train_test_split(X, Y, test_size=.2, random_state=5, stratify=Y)
