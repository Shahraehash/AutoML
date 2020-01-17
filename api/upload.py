"""
Upload dataset
"""

import os
import uuid

from flask import abort, jsonify, request

def upload(userid):
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

