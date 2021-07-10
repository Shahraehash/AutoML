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
        abort(400)
        return

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

    try:
        process_files(folder, request.form['label_column'])
    except ValueError as reason:
        rmtree(folder)
        abort(406, jsonify({
          'reason': reason
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

    return {
        'analysis': Describe(folder, metadata['label']),
        'label': metadata['label']
    }

def process_files(folder, label_column):
    """Cleans CSV headers and generates dataset metadata"""

    features = clean_csv_headers(folder + '/train.csv', label_column)
    clean_csv_headers(folder + '/test.csv', label_column)

    train = import_csv(folder + '/train.csv', label_column)[0]
    test = import_csv(folder + '/test.csv', label_column)[0]

    if train.shape[0] < 50:
        raise ValueError('training_rows_insufficient')

    if train.shape[0] > 20000:
        raise ValueError('training_rows_excess')

    if train.shape[1] > 5000:
        raise ValueError('training_features_excess')

    if test.shape[0] > 100000:
        raise ValueError('test_rows_excess')

    metadata = {
        'label': label_column,
        'features': features
    }
    with open(folder + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

def clean_csv_headers(file, label_column):
    """Strips spaces from CSV headers"""

    csv = pd.read_csv(file)
    csv.rename(columns=lambda x: x.strip(), inplace=True)
    csv.to_csv(file, index=False)
    return csv.columns.drop(label_column).tolist()
