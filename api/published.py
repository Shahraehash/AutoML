"""
Handle published model requests
"""

import ast
import os
import json

from flask import abort, jsonify, request, send_file

from ml.predict import predict

PUBLISHED_MODELS = 'data/published-models.json'

def get(userid):
    """Get all published models for a given user ID"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    user = userid.urn[9:]
    published = {
        k:{
            'features': ast.literal_eval(v['features']),
            'date': v['date']
        } for (k, v) in published.items() if user in v['path']
    }

    return jsonify(published)

def delete(name):
    """Removes a published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if name not in published:
        abort(400)
        return

    published.pop(name, None)

    with open(PUBLISHED_MODELS, 'w') as published_file:
        json.dump(published, published_file)

    return jsonify({'success': True})

def test(name):
    """Tests the published model against the provided data"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if name not in published:
        abort(400)
        return

    folder = published[name]['path'][:published[name]['path'].rfind('/')]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    reply = predict(
        json.loads(request.data),
        published[name]['path']
    )

    reply['target'] = metadata['label']

    return jsonify(reply)

def features(name):
    """Returns the features for a published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if name not in published:
        abort(400)
        return

    return jsonify(published[name]['features'])

def export_pmml(name):
    """Export the published model's PMML"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if name not in published:
        abort(400)
        return

    if not os.path.exists(published[name]['path'] + '.pmml'):
        abort(400)
        return

    return send_file(published[name]['path'] + '.pmml', as_attachment=True)


def export_model(name):
    """Export the published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if name not in published:
        abort(400)
        return

    if not os.path.exists(published[name]['path'] + '.joblib'):
        abort(400)
        return

    return send_file(published[name]['path'] + '.joblib', as_attachment=True)
