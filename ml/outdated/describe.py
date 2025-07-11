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
    
    # Get unique labels for multi-class support
    unique_labels = sorted(csv_clean[label].unique())
    
    # Create histograms for each class instead of just positive/negative
    histograms_by_class = {}
    for class_label in unique_labels:
        class_data = csv_clean[csv_clean[label] == class_label]
        histograms_by_class[f'class_{int(class_label)}'] = {
            key: [i.tolist() for i in np.histogram(
                class_data[key].values, 
                bins=10, 
                range=(csv_clean[key].min(), csv_clean[key].max())
            )] for key in class_data.columns
        }
    
    return {
        'null': len(csv.index) - len(csv_clean.index),
        'invalid': csv.loc[:, (csv.dtypes != np.int64) & (csv.dtypes != np.float64)].columns.values.tolist(),
        'mode': csv_clean.mode().iloc[0].to_dict(),
        'median': csv_clean.median().to_dict(),
        'summary': csv_clean.describe().to_dict(),
        'histogram': {
            'by_class': histograms_by_class
        }
    }
