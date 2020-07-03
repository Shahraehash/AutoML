"""
Search for the best model for a given dataset
"""

import ast
import os
import json
import time
import uuid
from shutil import copyfile, rmtree

from flask import abort, g, jsonify, request, send_file, url_for
from numpy.lib.financial import npv
import pandas as pd

from ml.create_model import create_model
from ml.list_pipelines import list_pipelines
from ml.predict import predict
from worker import queue_training

def get():
    """Get all the jobs for a given user ID"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs'

    if not os.path.exists(folder):
        abort(400)
        return

    jobs = []
    for job in os.listdir(folder):
        if not os.path.isdir(folder + '/' + job) or\
            not os.path.exists(folder + '/' + job + '/metadata.json'):
            continue

        with open(folder + '/' + job + '/metadata.json') as metafile:
            metadata = json.load(metafile)

        jobs.append({
            'date': time.strftime(
                '%Y-%m-%dT%H:%M:%SZ',
                time.gmtime(max(
                    os.path.getmtime(root) for root, _, _ in os.walk(folder + '/' + job)
                ))
            ),
            'id': job,
            'hasResults': os.path.exists(folder + '/' + job + '/report.csv'),
            'metadata': metadata
        })

    return jsonify(jobs)

def add():
    """Creates a new job"""

    if g.uid is None:
        abort(401)
        return

    try:
        datasetid = request.get_json()['datasetid']
    except KeyError:
        abort(400)
        return

    jobid = uuid.uuid4().urn[9:]

    folder = 'data/users/' + g.uid + '/jobs/' + jobid

    if not os.path.exists(folder):
        os.makedirs(folder)

    metadata = {}
    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    metadata['datasetid'] = datasetid

    with open(folder + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

    return jsonify({'id': jobid})

def delete(jobid):
    """Deletes a previous job"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    rmtree(folder)

    return jsonify({'success': True})

def train(jobid):
    """Finds the best model for the selected parameters/data"""

    if g.uid is None:
        abort(401)
        return

    parameters = request.form.to_dict()
    pipelines = list_pipelines(parameters)

    job_folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    task = queue_training.s(
        g.uid, jobid.urn[9:], dataset_metadata['label'], parameters
    ).apply_async()

    return jsonify({
        "id": task.id,
        "href": url_for('status', task_id=task.id),
        "pipelines": pipelines
    }), 202

def result(jobid):
    """Retrieve the training results"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    metadata = None

    if not os.path.exists(folder + '/report.csv'):
        abort(400)
        return

    try:
        data = json.loads(pd.read_csv(folder + '/report.csv').to_json(orient='records'))
    except ValueError:
        abort(400)

    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    return jsonify({
        'results': data,
        'metadata': metadata
    })

def get_pipelines(jobid):
    """Returns the pipelines for a job"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/' + '/metadata.json'):
        abort(400)
        return

    with open(folder + '/' + '/metadata.json') as json_file:
        metadata = json.load(json_file)

    return jsonify(list_pipelines(metadata['parameters']))

def refit(jobid):
    """Create a static copy of the selected model"""

    if g.uid is None:
        abort(401)
        return

    job_folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    generalization_result = create_model(
        request.form['key'],
        ast.literal_eval(request.form['parameters']),
        ast.literal_eval(request.form['features']),
        dataset_folder,
        dataset_metadata['label'],
        job_folder
    )

    return jsonify({'generalization': generalization_result})

def tandem(jobid):
    """Create a static copy of the two selected models to be used in tandem"""

    if g.uid is None:
        abort(401)
        return

    job_folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    npv_generalization_result = create_model(
        request.form['npv_key'],
        ast.literal_eval(request.form['npv_parameters']),
        ast.literal_eval(request.form['npv_features']),
        dataset_folder,
        dataset_metadata['label'],
        job_folder
    )

    with open(job_folder + '/tandem_npv_features.json', 'w') as npv_features:
        json.dump(ast.literal_eval(request.form['npv_features']), npv_features)

    copyfile(job_folder + '/pipeline.joblib', job_folder + '/tandem_npv.joblib')
    copyfile(job_folder + '/pipeline.pmml', job_folder + '/tandem_npv.pmml')
    copyfile(job_folder + '/pipeline.json', job_folder + '/tandem_npv.json')

    ppv_generalization_result = create_model(
        request.form['ppv_key'],
        ast.literal_eval(request.form['ppv_parameters']),
        ast.literal_eval(request.form['ppv_features']),
        dataset_folder,
        dataset_metadata['label'],
        job_folder
    )

    with open(job_folder + '/tandem_ppv_features.json', 'w') as ppv_features:
        json.dump(ast.literal_eval(request.form['ppv_features']), ppv_features)

    copyfile(job_folder + '/pipeline.joblib', job_folder + '/tandem_ppv.joblib')
    copyfile(job_folder + '/pipeline.pmml', job_folder + '/tandem_ppv.pmml')
    copyfile(job_folder + '/pipeline.json', job_folder + '/tandem_ppv.json')

    return jsonify({
        'npv_generalization': npv_generalization_result,
        'ppv_generalization_result': ppv_generalization_result
    })

def test(jobid):
    """Tests the selected model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    reply = predict(
        json.loads(request.data),
        folder + '/pipeline'
    )

    reply['target'] = metadata['label']

    return jsonify(reply)

def test_tandem(jobid):
    """Tests the selected tandem model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    with open(folder + '/tandem_npv_features.json') as feature_file:
        npv_features = json.load(feature_file)

    payload = json.loads(request.data)
    data = pd.DataFrame(payload['data'], columns=payload['features'])

    npv_reply = pd.DataFrame(predict(
        data[npv_features].to_numpy(),
        folder + '/tandem_npv'
    ))

    with open(folder + '/tandem_ppv_features.json') as feature_file:
        ppv_features = json.load(feature_file)

    ppv_reply = pd.DataFrame(predict(
        data[ppv_features].to_numpy(),
        folder + '/tandem_ppv'
    ))

    ppv_reply['predicted'] = ppv_reply.apply(lambda row: row['predicted'] if row['predicted'] > 0 else -1, axis=1)
    npv_reply['predicted'] = npv_reply.apply(lambda row: ppv_reply.iloc[row.name]['predicted'] if row['predicted'] > 0 else row['predicted'], axis=1)

    return jsonify({
      'predicted': npv_reply['predicted'].to_list(),
      'probability': npv_reply['probability'].to_list(),
      'target': metadata['label']
    })

def export(jobid):
    """Export the results CSV"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/report.csv'):
        abort(400)
        return

    return send_file(folder + '/report.csv', as_attachment=True, cache_timeout=-1)

def export_pmml(jobid):
    """Export the selected model's PMML"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.pmml'):
        abort(400)
        return

    return send_file(folder + '/pipeline.pmml', as_attachment=True, cache_timeout=-1)

def export_model(jobid):
    """Export the selected model"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.joblib'):
        abort(400)
        return

    return send_file(folder + '/pipeline.joblib', as_attachment=True, cache_timeout=-1)

def star_models(jobid):
    """Marks the selected models as starred"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if os.path.exists(folder + '/starred.json'):
        with open(folder + '/starred.json') as starred_file:
            starred = json.load(starred_file)
    else:
        starred = []

    starred = list(set(starred + request.get_json()['models']))

    with open(folder + '/starred.json', 'w') as starred_file:
        json.dump(starred, starred_file)

    return jsonify({'success': True})

def un_star_models(jobid):
    """Removes the selected models as starred models"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/starred.json'):
        return jsonify({'success': True})

    with open(folder + '/starred.json') as starred_file:
        starred = json.load(starred_file)

    for item in request.get_json()['models']:
        try:
            starred.remove(item)
        except Exception:
            continue

    with open(folder + '/starred.json', 'w') as starred_file:
        json.dump(starred, starred_file)

    return jsonify({'success': True})

def get_starred(jobid):
    """Get the starred models for a job"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/starred.json'):
        return abort(404)

    with open(folder + '/starred.json') as starred_file:
        starred = json.load(starred_file)

    return jsonify(starred)
