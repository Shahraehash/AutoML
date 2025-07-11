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

def import_csv(path, label_column, show_warning=False):
    """Import the specificed sheet"""

    # Read the CSV to memory and drop rows with empty values
    data = pd.read_csv(path)

    # Convert cell values to numerical data and drop invalid data
    data = data.apply(pd.to_numeric, errors='coerce').dropna()

    # Drop the label column from the data
    x = data.drop(label_column, axis=1)

    # Save the label colum values
    y = data[label_column]

    # Grab the feature names
    feature_names = list(x)

    # Convert to NumPy array
    x = x.to_numpy()

    # Get unique labels and label counts
    unique_labels = sorted(y.unique())

    label_counts = {}
    for label in unique_labels:
        label_counts[f'class_{int(label)}_count'] = data[data[label_column] == label].shape[0]
    
    # For backward compatibility with binary classification
    if len(unique_labels) == 2:
        if show_warning:
            negative_count = label_counts.get('class_0_count', 0)
            positive_count = label_counts.get('class_1_count', 0)
            print('Negative Cases: %.7g\nPositive Cases: %.7g\n' % (negative_count, positive_count))
            if negative_count / positive_count < .9:
                print('Warning: Classes are not balanced.')
                
        return [x, y, feature_names, label_counts, 2]
        
    else:
        # Multi-class case
        if show_warning:
            for label in unique_labels:
                count = label_counts[f'class_{int(label)}_count']
                print('Class %d Cases: %.7g\n' % (int(label), count))

            # Check for class imbalance in multi-class
            counts = list(label_counts.values())
            min_count, max_count = min(counts), max(counts)
            if min_count / max_count < .5: #NOT SURE WHAT THIS THRESHOLD SHOULD BE?
                print('Warning: Classes are not balanced.')
        
        # Return all class counts as a dictionary for multi-class
        return [x, y, feature_names, label_counts, len(unique_labels)]
