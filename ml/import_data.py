"""
Import data and process/clean data
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def normalize_labels(y, custom_labels=None):
    """
    Normalize class labels to consecutive integers starting from 0.
    Returns normalized labels and the mapping dictionary including custom labels.
    """
    unique_labels = sorted(y.unique())
    
    # Check if labels are already consecutive starting from 0
    expected_labels = list(range(len(unique_labels)))
    if unique_labels == expected_labels:
        # Labels are already normalized
        mapping_info = None
    else:
        # Create mapping from original labels to consecutive labels
        # Ensure all keys and values are native Python types for JSON serialization
        label_mapping = {int(original.item()) if hasattr(original, 'item') else int(original): new 
                        for new, original in enumerate(unique_labels)}
        inverse_mapping = {new: int(original.item()) if hasattr(original, 'item') else int(original) 
                          for original, new in label_mapping.items()}
        
        # Transform labels
        y = y.map(label_mapping)
        mapping_info = {
            'label_mapping': label_mapping,
            'inverse_mapping': inverse_mapping,
            'original_labels': [int(label.item()) if hasattr(label, 'item') else int(label) for label in unique_labels]
        }
    
    # Add custom labels to mapping info
    if custom_labels and mapping_info:
        # Map custom labels to normalized indices
        custom_label_mapping = {}
        for original_label_str, custom_label in custom_labels.items():
            try:
                # Handle float strings like "0.0" by converting to float first, then int
                original_int = int(float(original_label_str))
                if original_int in mapping_info['label_mapping']:
                    normalized_index = mapping_info['label_mapping'][original_int]
                    custom_label_mapping[normalized_index] = custom_label
                else:
                    print(f"WARNING: Original label {original_int} not found in label mapping {list(mapping_info['label_mapping'].keys())}")
            except (ValueError, TypeError) as e:
                print(f"ERROR: Could not process custom label '{original_label_str}': {e}")
        
        mapping_info['custom_labels'] = custom_label_mapping
    elif custom_labels:
        # Labels were already normalized, create direct mapping
        # Handle float strings like "0.0" by converting to float first, then int
        custom_label_mapping = {}
        for k, v in custom_labels.items():
            try:
                int_key = int(float(k))
                custom_label_mapping[int_key] = v
            except (ValueError, TypeError) as e:
                print(f"ERROR: Could not process direct custom label '{k}': {e}")
        
        mapping_info = {
            'custom_labels': custom_label_mapping
        }
    return y, mapping_info

def import_data(train_path, test_path, label_column):
    """Import both the training and test data using the passed paths"""

    x, y, feature_names, train_class_counts, num_classes, train_label_mapping = import_csv(train_path, label_column, True)
    x2, y2, _, test_class_counts, _, test_label_mapping = import_csv(test_path, label_column)

    # Use train label mapping as the primary mapping (test should have same classes)
    label_mapping_info = train_label_mapping or test_label_mapping

    metadata = {
        'train_class_counts': train_class_counts,
        'test_class_counts': test_class_counts,
        'num_classes': num_classes,
        'label_mapping': label_mapping_info
    }

    return train_test_split(x, y, test_size=.2, random_state=5, stratify=y) + \
        [x2, y2, feature_names, metadata]

def import_train(train_path, label_column):
    """Import training data using the passed path"""

    x, y, feature_names, _, _, label_mapping_info = import_csv(train_path, label_column, True)
    return train_test_split(x, y, test_size=.2, random_state=5, stratify=y) + [feature_names, label_mapping_info]

def import_csv(path, label_column, show_warning=False, custom_labels=None):
    """Import the specificed sheet"""

    # Read the CSV to memory and drop rows with empty values
    data = pd.read_csv(path)

    # Convert cell values to numerical data and drop invalid data
    data = data.apply(pd.to_numeric, errors='coerce').dropna()

    # Drop the label column from the data
    x = data.drop(label_column, axis=1)

    # Save the label colum values
    y = data[label_column]

    # Normalize labels to consecutive integers starting from 0
    y_normalized, label_mapping_info = normalize_labels(y, custom_labels)

    # Grab the feature names
    feature_names = list(x)

    # Convert to NumPy array
    x = x.to_numpy()

    # Get unique labels (from original data for counts, but use normalized for processing)
    original_unique_labels = sorted(y.unique())
    normalized_unique_labels = sorted(y_normalized.unique())

    # Calculate label counts using original labels for reporting
    # Ensure label keys are native Python types for JSON serialization
    label_counts = {}
    for label in original_unique_labels:
        # Convert numpy int64 to native Python int
        python_label = int(label.item()) if hasattr(label, 'item') else int(label)
        label_counts[f'class_{python_label}_count'] = int(data[data[label_column] == label].shape[0])
    
    # Add label mapping info to the return data
    # Ensure num_classes is native Python int for JSON serialization
    num_classes = int(len(original_unique_labels))
    
    # For backward compatibility with binary classification
    if num_classes == 2:
        if show_warning:
            # Convert numpy int64 to native Python int for dictionary key lookup
            label_0 = int(original_unique_labels[0].item()) if hasattr(original_unique_labels[0], 'item') else int(original_unique_labels[0])
            label_1 = int(original_unique_labels[1].item()) if hasattr(original_unique_labels[1], 'item') else int(original_unique_labels[1])
            negative_count = label_counts.get(f'class_{label_0}_count', 0)
            positive_count = label_counts.get(f'class_{label_1}_count', 0)
            print('Negative Cases: %.7g\nPositive Cases: %.7g\n' % (negative_count, positive_count))
            if negative_count / positive_count < .9:
                print('Warning: Classes are not balanced.')
                
        return [x, y_normalized, feature_names, label_counts, num_classes, label_mapping_info]
        
    else:
        # Multi-class case
        if show_warning:
            for label in original_unique_labels:
                # Convert numpy int64 to native Python int
                python_label = int(label.item()) if hasattr(label, 'item') else int(label)
                count = label_counts[f'class_{python_label}_count']
                print('Class %d Cases: %.7g\n' % (python_label, count))

            # Check for class imbalance in multi-class
            counts = list(label_counts.values())
            min_count, max_count = min(counts), max(counts)
            if min_count / max_count < .5: #NOT SURE WHAT THIS THRESHOLD SHOULD BE?
                print('Warning: Classes are not balanced.')
        
        # Return all class counts as a dictionary for multi-class
        return [x, y_normalized, feature_names, label_counts, num_classes, label_mapping_info]
