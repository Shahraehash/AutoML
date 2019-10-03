"""
Import data and process/clean data
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def import_data(train_path, test_path, label_column):
    """Import data using the passed paths"""

    data = pd.read_csv(train_path).dropna()
    x = data.drop(label_column, axis=1)
    y = data[label_column]

    data_test = pd.read_csv(test_path).dropna()
    x2 = data_test.drop(label_column, axis=1)
    y2 = data_test[label_column]

    # Only keep numeric inputs
    x = x.loc[:, (x.dtypes == np.int64) | (x.dtypes == np.float64)].dropna()
    x2 = x2.loc[:, (x2.dtypes == np.int64) | (x2.dtypes == np.float64)].dropna()

    feature_names = list(x)

    x = x.to_numpy()
    x2 = x2.to_numpy()

    negative_count = data[data[label_column] == 0].shape[0]
    positive_count = data[data[label_column] == 1].shape[0]

    print('Negative Cases: %.7g\nPositive Cases: %.7g\n' % (negative_count, positive_count))

    if negative_count / positive_count < .9:
        print('Warning: Classes are not balanced.')

    # Generate test/train split from the train data
    return train_test_split(x, y, test_size=.2, random_state=5, stratify=y) + [x2, y2, feature_names]
