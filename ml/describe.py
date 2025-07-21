"""
Generate descriptive statistics for a provided
dataset.
"""

import pandas as pd
import numpy as np

def describe(folder, label, custom_labels=None):
    """Accept the training and testing data sets and describe them"""

    train_analysis = parse_csv(folder + '/train.csv', label, custom_labels)
    test_analysis = parse_csv(folder + '/test.csv', label, custom_labels)
    
    return {
        'train': train_analysis,
        'test': test_analysis,
        'n_classes': train_analysis['n_classes'],
        'class_type': 'multiclass' if train_analysis['n_classes'] > 2 else 'binary'
    }

def sanitize_class_key(class_name):
    """
    Sanitizes class names to be safe for use as dictionary keys and CSS selectors
    while preserving readability
    """
    return class_name.replace(' ', '_').replace('-', '_').lower()

def parse_csv(csv_file, label, custom_labels=None):
    """Parse the CSV file and get required details"""

    csv = pd.read_csv(csv_file)
    csv_clean = csv.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Get unique labels for multi-class support
    unique_labels = sorted(csv_clean[label].unique())
    
    # Create histograms for each class instead of just positive/negative
    histograms_by_class = {}
    class_label_mapping = {}  # Track original labels for display
    
    for class_label in unique_labels:
        class_data = csv_clean[csv_clean[label] == class_label]
        
        # Convert class_label to string for consistent key lookup
        class_key = str(int(class_label))
        
        # Use custom label if available, otherwise fall back to generic class name
        if custom_labels and class_key in custom_labels:
            display_label = custom_labels[class_key]
        else:
            display_label = f'Class {int(class_label)}'
        
        # Create a safe key for the histogram data
        safe_key = sanitize_class_key(display_label)
        
        # Store the mapping for frontend use
        class_label_mapping[safe_key] = display_label
        
        histograms_by_class[safe_key] = {
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
            'by_class': histograms_by_class,
            'class_label_mapping': class_label_mapping
        },
        'n_classes': len(unique_labels)
    }
