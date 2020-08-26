"""
Generate descriptive statistics for a provided
dataset.
"""

import pandas as pd
import numpy as np

def describe(folder, label):
    """Accept the training and testing data sets and describe them"""

    return {
        'train': parse_csv(folder + '/train.csv', label),
        'test': parse_csv(folder + '/test.csv', label)
    }

def parse_csv(csv_file, label):
    """Parse the CSV file and get required details"""

    csv = pd.read_csv(csv_file)
    csv_clean = csv.apply(pd.to_numeric, errors='coerce').dropna()
    histogram = {key:[i.tolist() for i in np.histogram(list(value.values()), bins='auto', range=(1,100))] for (key,value) in csv_clean.to_dict().items()}
    histogram[label] = [i.tolist() for i in np.histogram(csv_clean[label].to_list(), bins='auto')]

    return {
        'null': len(csv.index) - len(csv_clean.index),
        'invalid': csv.loc[:, (csv.dtypes != np.int64) & (csv.dtypes != np.float64)].columns.values.tolist(),
        'mode': csv_clean.mode().iloc[0].to_dict(),
        'median': csv_clean.median().to_dict(),
        'summary': csv_clean.describe().to_dict(),
        'histogram': histogram
    }
