"""
Tests a published model
"""

import os
import json

from flask import abort, jsonify, request

from ml.predict import predict

PUBLISHED_MODELS = 'data/published-models.json'

def test_published_model(model):
    """Tests the published model against the provided data"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(400)
        return

    folder = published[model]['path'][:published[model]['path'].rfind('/')]
    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    reply = predict(
        json.loads(request.data),
        published[model]['path']
    )

    reply['target'] = label_column

    return jsonify(reply)

def test_model(userid, jobid):
    """Tests the selected model against the provided data"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    reply = predict(
        json.loads(request.data),
        folder + '/pipeline'
    )

    reply['target'] = label_column

    return jsonify(reply)