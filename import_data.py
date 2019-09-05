# Dependencies
import pandas as pd

from sklearn.model_selection import train_test_split

def importData(trainPath, testPath, labelColumn):
    data = pd.read_csv(trainPath).dropna()
    X = data.drop(labelColumn, axis=1)
    Y = data[labelColumn]

    data_test = pd.read_csv(testPath).dropna()
    X2 = data_test.drop(labelColumn, axis=1)
    Y2 = data_test[labelColumn]

    # Generate test/train split from the train data
    return [data, data_test, X, Y, X2, Y2] + train_test_split(X, Y, test_size=.2, random_state=5, stratify=Y)

