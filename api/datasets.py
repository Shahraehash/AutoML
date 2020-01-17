"""
Handle dataset related requests
"""

import os
import uuid

from flask import abort, jsonify, request

from ml.describe import describe as Describe

def get(userid):
    """Get all the datasets for a given user ID"""

    folder = 'data/' + userid.urn[9:] + '/datasets'

    if not os.path.exists(folder):
        abort(400)
        return

    jobs = []
    for job in os.listdir(folder):
        if not os.path.isdir(folder + '/' + job) or\
            not os.path.exists(folder + '/' + job + '/train.csv') or\
            not os.path.exists(folder + '/' + job + '/test.csv') or\
            not os.path.exists(folder + '/' + job + '/label.txt'):
            continue

        label = open(folder + '/' + job + '/label.txt', 'r')
        label_column = label.read()
        label.close()

        jobs.append({
            'id': job,
            'label': label_column
        })

    return jsonify(jobs)

def add(userid):
    """Upload files to the server"""

    if 'train' not in request.files or 'test' not in request.files:
        return abort(400)

    train = request.files['train']
    test = request.files['test']

    datasetid = uuid.uuid4().urn[9:]

    folder = 'data/' + userid.urn[9:] + '/datasets/' + datasetid

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

def describe(userid, datasetid):
    """Generate descriptive statistics for training/testing datasets"""

    folder = 'data/' + userid.urn[9:] + '/datasets/' + datasetid.urn[9:]

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
