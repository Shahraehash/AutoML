"""
Handle dataset related requests
"""

import os
import json
import time
import uuid
from shutil import rmtree

import pandas as pd
from flask import abort, g, jsonify, request

from ml.describe import describe as Describe
from ml.import_data import import_csv

def get():
    """Get all the datasets for a given user ID"""

    if g.uid is None:
        abort(401)
        return

    datasets = []
    folder = 'data/users/' + g.uid + '/datasets'

    if not os.path.exists(folder):
        # Create the user datasets folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        return jsonify(datasets)

    for dataset in os.listdir(folder):
        if not os.path.isdir(folder + '/' + dataset) or\
            not os.path.exists(folder + '/' + dataset + '/train.csv') or\
            not os.path.exists(folder + '/' + dataset + '/test.csv') or\
            not os.path.exists(folder + '/' + dataset + '/metadata.json'):
            continue

        with open(folder + '/' + dataset + '/metadata.json') as metafile:
            dataset_metadata = json.load(metafile)

        datasets.append({
            'date': time.strftime(
                '%Y-%m-%dT%H:%M:%SZ',
                time.gmtime(max(
                    os.path.getmtime(root) for root, _, _ in os.walk(folder + '/' + dataset)
                ))
            ),
            'id': dataset,
            'label': dataset_metadata['label'],
            'features': dataset_metadata['features']
        })

    return jsonify(datasets)

def add():
    """Upload files to the server"""

    if g.uid is None:
        abort(401)
        return

    if 'train' not in request.files or 'test' not in request.files:
        return abort(400)

    train = request.files['train']
    test = request.files['test']

    datasetid = uuid.uuid4().urn[9:]

    folder = 'data/users/' + g.uid + '/datasets/' + datasetid

    if not os.path.exists(folder):
        os.makedirs(folder)

    if not train or not test:
        abort(400)
        return

    train.save(folder + '/train.csv')
    test.save(folder + '/test.csv')

    # Extract class labels if provided
    class_labels = None
    if 'class_labels' in request.form:
        class_labels = json.loads(request.form['class_labels'])
        

    try:
        process_files(folder, request.form['label_column'], class_labels)
    except ValueError as reason:
        print(f"Dataset validation failed for user {g.uid}: {reason}")
        rmtree(folder)
        abort(406, jsonify({
          'reason': str(reason)
        }))
    except Exception as e:
        print(f"Unexpected error processing dataset for user {g.uid}: {e}")
        rmtree(folder)
        abort(500, jsonify({
          'error': str(e)
        }))

    return jsonify({'id': datasetid})

def delete(datasetid):
    """Deletes a dataset"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/datasets/' + datasetid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    rmtree(folder)

    return jsonify({'success': True})

def describe(datasetid):
    """Generate descriptive statistics for training/testing datasets"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/datasets/' + datasetid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    # Extract custom class labels if they exist
    custom_labels = metadata.get('class_labels', None)

    return {
        'analysis': Describe(folder, metadata['label'], custom_labels),
        'label': metadata['label']
    }

def process_files(folder, label_column, class_labels=None):
    """Cleans CSV headers and generates dataset metadata"""

    features = clean_csv_headers(folder + '/train.csv', label_column)
    clean_csv_headers(folder + '/test.csv', label_column)

    # Pass custom labels to import_csv for processing and get label mapping info
    train_data, train_y, train_features, train_counts, train_num_classes, train_label_mapping = import_csv(
        folder + '/train.csv', label_column, custom_labels=class_labels
    )
    test_data, test_y, test_features, test_counts, test_num_classes, test_label_mapping = import_csv(
        folder + '/test.csv', label_column, custom_labels=class_labels
    )

    if train_data.shape[0] < 50:
        raise ValueError('training_rows_insufficient')

    if train_data.shape[0] > 10000:
        raise ValueError('training_rows_excess')

    if train_data.shape[1] > 2000:
        raise ValueError('training_features_excess')

    if test_data.shape[0] > 100000:
        raise ValueError('test_rows_excess')

    # Use train label mapping as primary (test should have same classes)
    label_mapping_info = train_label_mapping or test_label_mapping

    metadata = {
        'label': label_column,
        'features': features,
        'class_labels': class_labels,  # Store original custom class labels
        'label_mapping': label_mapping_info  # Store the complete label mapping info
    }
    with open(folder + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

def clean_csv_headers(file, label_column):
    """Strips spaces from CSV headers"""

    csv = pd.read_csv(file)
    csv.rename(columns=lambda x: x.strip(), inplace=True)
    csv.to_csv(file, index=False)
    return csv.columns.drop(label_column).tolist()
