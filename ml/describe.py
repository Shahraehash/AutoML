"""
Generate descriptive statistics for a provided
dataset.
"""

import pandas as pd
import numpy as np

def describe(folder):
    """Accept the training and testing data sets and describe them"""

    return {
        'train': parse_csv(folder + '/train.csv'),
        'test': parse_csv(folder + '/test.csv')
    }

def parse_csv(csv_file):
    """Parse the CSV file and get required details"""

    csv = pd.read_csv(csv_file)
    csv_clean = csv.loc[:, (csv.dtypes == np.int64) | (csv.dtypes == np.float64)].dropna()

    return {
        'null': csv.isnull().sum(axis=0).to_dict(),
        'invalid': csv.loc[:, (csv.dtypes != np.int64) & (csv.dtypes != np.float64)].columns.values.tolist(),
        'mode': csv_clean.mode().iloc[0].to_dict(),
        'median': csv_clean.median().to_dict(),
        'summary': csv_clean.describe().to_dict(),
        'histogram': {key:[i.tolist() for i in np.histogram(list(value.values()), bins='fd')] for (key,value) in csv_clean.to_dict().items()}
    }
