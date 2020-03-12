"""
Handle published model requests
"""

import ast
import os
import json
import time
import uuid
from shutil import copyfile

from flask import abort, g, jsonify, request, send_file

from ml.predict import predict
from .jobs import refit

PUBLISHED_MODELS = 'data/published-models.json'

def get():
    """Get all published models for a given user ID"""

    if g.uid is None:
        abort(401)
        return

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    published = {
        k:{
            'features': ast.literal_eval(v['features']),
            'date': v['date']
        } for (k, v) in published.items() if g.uid in v['path']
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

def rename(name):
    """Renames a published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if name not in published:
        abort(400)
        return

    published[request.get_json()['name']] = published.pop(name)

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

    if not os.path.exists(folder + '/metadata.json'):
        abort(400)
        return

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

    if 'restricted' in published[name] and g.uid not in published[name]['restricted']:
        abort(400)
        return

    with open(published[name]['path'] + '.json') as statsfile:
        generalization = json.load(statsfile)

    return jsonify({
        'features': published[name]['features'],
        'generalization': generalization
    })

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

    return send_file(published[name]['path'] + '.pmml', as_attachment=True, cache_timeout=-1)


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

    return send_file(published[name]['path'] + '.joblib', as_attachment=True, cache_timeout=-1)

def add(name):
    """Refits a model for future use via a published name"""

    if g.uid is None:
        abort(401)
        return

    refit(uuid.UUID(request.form['job']))

    job_folder = 'data/users/' + g.uid + '/jobs/' + request.form['job']
    model_path = job_folder + '/' + name

    copyfile(job_folder + '/pipeline.joblib', model_path + '.joblib')
    copyfile(job_folder + '/pipeline.pmml', model_path + '.pmml')
    copyfile(job_folder + '/pipeline.json', model_path + '.json')

    if os.path.exists(PUBLISHED_MODELS):
        with open(PUBLISHED_MODELS) as published_file:
            published = json.load(published_file)
    else:
        published = {}

    if name in published:
        abort(409)
        return

    published[name] = {
        'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'features': request.form['features'],
        'path': model_path
    }

    with open(PUBLISHED_MODELS, 'w') as published_file:
        json.dump(published, published_file)

    return jsonify({'success': True})
