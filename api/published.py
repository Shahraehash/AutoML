"""
Handle published model requests
"""

import ast
import os
import re
import json
import time
import uuid
import zipfile
from io import BytesIO
from shutil import copyfile

from flask import abort, g, jsonify, request, send_file

from ml.unified_classifier_manager import UnifiedClassifierManager
from .jobs import refit

PUBLISHED_MODELS = 'data/published-models.json'

def get():
    """Get all published models for a given user ID"""

    if g.uid is None:
        abort(401)
        return

    if not os.path.exists(PUBLISHED_MODELS):
        # Create the published models file if it doesn't exist
        with open(PUBLISHED_MODELS, 'w') as published_file:
            json.dump({}, published_file)
        return jsonify({})

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

    # Create UnifiedClassifierManager and load static model
    manager = UnifiedClassifierManager(folder, metadata.get('parameters', {}))
    
    # Load the static published model
    success = manager.load_static_model(published[name]['path'] + '.joblib')
    
    if success:
        # Use the new static model prediction method
        reply = manager.make_predictions(
            json.loads(request.data),
            'published_model',
            published[name]['threshold']
        )
        reply['target'] = metadata['label']
        return jsonify(reply)
    else:
        raise Exception("Failed to load static published model")

def generalize(name):
    """Generalize the published model against the provided data"""

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

    # Create UnifiedClassifierManager and load static model
    manager = UnifiedClassifierManager(folder, metadata.get('parameters', {}))
    
    # Load the static published model
    success = manager.load_static_model(published[name]['path'] + '.joblib')
    
    if success:
        # Use the new static model evaluation methods
        return jsonify({
            'generalization': manager.get_static_generalization_metrics(json.loads(request.data), 'published_model', metadata['label'], published[name]['threshold']),
            'reliability': manager.get_static_reliability_metrics(json.loads(request.data), 'published_model', metadata['label']),
            'precision_recall': manager.get_static_precision_metrics(json.loads(request.data), 'published_model', metadata['label']),
            'roc_auc': manager.get_static_roc_metrics(json.loads(request.data), 'published_model', metadata['label'])
        })
    else:
        raise Exception("Failed to load static published model")

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
        'feature_scores': published[name]['feature_scores'],
        'generalization': generalization,
        'threshold': published[name]['threshold'] if 'threshold' in published[name] else .5
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

    try:
        copyfile(published[name]['path'] + '.csv', 'client/input.csv')
    except Exception:
        return send_file(published[name]['path'] + '.joblib', as_attachment=True, cache_timeout=-1)

    threshold = published[name]['threshold'] if 'threshold' in published[name] else .5
    with open('client/predict.py', 'r+') as file:
        contents = file.read()
        contents = re.sub(r'THRESHOLD = [\d.]+', 'THRESHOLD = ' + str(threshold), contents)
        file.seek(0)
        file.truncate()
        file.write(contents)

    copyfile(published[name]['path'] + '.joblib', 'client/pipeline.joblib')

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        files = os.listdir('client')
        for individualFile in files:
            filePath = os.path.join('client', individualFile)
            zf.write(filePath, individualFile)
    memory_file.seek(0)

    return send_file(memory_file, attachment_filename=(name + '.zip'), as_attachment=True, cache_timeout=-1)

def add(name):
    """Refits a model for future use via a published name"""

    if g.uid is None:
        abort(401)
        return

    threshold = float(request.form['threshold'])

    refit(uuid.UUID(request.form['job']), threshold)

    job_folder = 'data/users/' + g.uid + '/jobs/' + request.form['job']
    model_path = job_folder + '/' + name

    copyfile(job_folder + '/pipeline.joblib', model_path + '.joblib')
    copyfile(job_folder + '/input.csv', model_path + '.csv')
    try:
        copyfile(job_folder + '/pipeline.pmml', model_path + '.pmml')
    except:
        pass
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
        'path': model_path,
        'threshold': threshold,
        'feature_scores': request.form['feature_scores']
    }

    with open(PUBLISHED_MODELS, 'w') as published_file:
        json.dump(published, published_file)

    return jsonify({'success': True})
