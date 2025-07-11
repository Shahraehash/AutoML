"""
Import data and process/clean data
"""

import pandas as pd

from sklearn.model_selection import train_test_split

def import_data(train_path, test_path, label_column):
    """Import both the training and test data using the passed paths"""

    x, y, feature_names, train_class_counts, num_classes = import_csv(train_path, label_column, True)
    x_test, y_test, _, test_class_counts, _ = import_csv(test_path, label_column)

    metadata = {
        'train_class_counts': train_class_counts,
        'test_class_counts': test_class_counts,
        'num_classes': num_classes
    }

    # Split training data into train (80%) and validation (20%)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=5, stratify=y)
    
    return x_train, x_val, y_train, y_val, x_test, y_test, feature_names, metadata

def import_train(train_path, label_column):
    """Import training data using the passed path"""

    x, y, feature_names, _, _ = import_csv(train_path, label_column, True)
    return train_test_split(x, y, test_size=.2, random_state=5, stratify=y) + [feature_names]

