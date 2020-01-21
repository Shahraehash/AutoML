"""
Handle dataset related requests
"""

import os
import time
import uuid
from shutil import rmtree

from flask import abort, jsonify, request

from ml.describe import describe as Describe

def get(userid):
    """Get all the datasets for a given user ID"""

    folder = 'data/users/' + userid + '/datasets'

    if not os.path.exists(folder):
        abort(400)
        return

    datasets = []
    for dataset in os.listdir(folder):
        if not os.path.isdir(folder + '/' + dataset) or\
            not os.path.exists(folder + '/' + dataset + '/train.csv') or\
            not os.path.exists(folder + '/' + dataset + '/test.csv') or\
            not os.path.exists(folder + '/' + dataset + '/label.txt'):
            continue

        label = open(folder + '/' + dataset + '/label.txt', 'r')
        label_column = label.read()
        label.close()

        datasets.append({
            'date': time.strftime(
                '%Y-%m-%dT%H:%M:%SZ',
                time.gmtime(max(
                    os.path.getmtime(root) for root, _, _ in os.walk(folder + '/' + dataset)
                ))
            ),
            'id': dataset,
            'label': label_column
        })

    return jsonify(datasets)

def add(userid):
    """Upload files to the server"""

    if 'train' not in request.files or 'test' not in request.files:
        return abort(400)

    train = request.files['train']
    test = request.files['test']

    datasetid = uuid.uuid4().urn[9:]

    folder = 'data/users/' + userid + '/datasets/' + datasetid

    if not os.path.exists(folder):
        os.makedirs(folder)

    if train and test:
        train.save(folder + '/train.csv')
        test.save(folder + '/test.csv')

        label = open(folder + '/label.txt', 'w')
        label.write(request.form['label_column'])
        label.close()

        return jsonify({'id': datasetid})

    return abort(400)

def delete(userid, datasetid):
    """Deletes a dataset"""

    folder = 'data/users/' + userid + '/datasets/' + datasetid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    rmtree(folder)

    return jsonify({'success': True})

def describe(userid, datasetid):
    """Generate descriptive statistics for training/testing datasets"""

    folder = 'data/users/' + userid + '/datasets/' + datasetid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    return {
        'analysis': Describe(folder),
        'label': label_column
    }
