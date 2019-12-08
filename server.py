"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

import ast
import os
import json
from shutil import copyfile

from celery import Celery
from flask import abort, Flask, jsonify, request, send_file, send_from_directory, url_for
from flask_cors import CORS
import pandas as pd

from api import api, create_model, predict

PUBLISHED_MODELS = 'data/published-models.json'

APP = Flask(__name__, static_url_path='')
CORS(APP)

CELERY = Celery(APP.name, backend='rpc://', broker='pyamqp://guest@localhost//')
CELERY.conf.update(task_track_started=True)

@CELERY.task()
def queue_training(userid, jobid, parameters):
    folder = 'data/' + userid + '/' + jobid

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    labels = ['No ' + label_column, label_column]

    os.environ['IGNORE_ESTIMATOR'] = parameters['ignore_estimator']
    os.environ['IGNORE_FEATURE_SELECTOR'] = parameters['ignore_feature_selector']
    os.environ['IGNORE_SCALER'] = parameters['ignore_scaler']
    os.environ['IGNORE_SEARCHER'] = parameters['ignore_searcher']
    os.environ['IGNORE_SCORER'] = parameters['ignore_scorer']
    if 'ignore_shuffle' in parameters:
        os.environ['IGNORE_SHUFFLE'] = parameters['ignore_shuffle']

    metadata = {}
    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    metadata['parameters'] = parameters

    with open(folder + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

    api.find_best_model(folder + '/train.csv', folder + '/test.csv', labels, label_column, folder)
    return {}

@APP.route('/')
def load_ui():
    """Loads `index.html` for the root path"""

    return send_from_directory('static', 'index.html')

@APP.route('/create/<uuid:userid>/<uuid:jobid>', methods=['POST'])
def create(userid, jobid):
    """Create a static copy of the selected model"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    create_model.create_model(
        request.form['key'],
        ast.literal_eval(request.form['parameters']),
        ast.literal_eval(request.form['features']),
        folder + '/train.csv',
        label_column,
        folder
    )

    if 'publishName' in request.form:
        model_path = folder + '/' + request.form['publishName']
        copyfile(folder + '/pipeline.joblib', model_path + '.joblib')
        copyfile(folder + '/pipeline.pmml', model_path + '.pmml')

        if os.path.exists(PUBLISHED_MODELS):
            with open(PUBLISHED_MODELS) as published_file:
                published = json.load(published_file)
        else:
            published = {}

        if request.form['publishName'] in published:
            abort(409)
            return

        published[request.form['publishName']] = {
            'features': request.form['features'],
            'path': model_path
        }

        with open(PUBLISHED_MODELS, 'w') as published_file:
            json.dump(published, published_file)

    return jsonify({'success': True})

@APP.route('/features/<string:model>', methods=['GET'])
def get_model_features(model):
    """Returns the features for a published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(404)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(404)
        return

    return jsonify(published[model]['features'])

@APP.route('/test/<string:model>', methods=['POST'])
def test_published_model(model):
    """Tests the published model against the provided data"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(404)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(404)
        return

    folder = published[model]['path'][:published[model]['path'].rfind('/')]
    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    reply = predict.predict(
        [float(x) for x in request.form['data'].split(',')],
        published[model]['path']
    )

    reply['target'] = label_column

    return jsonify(reply)

@APP.route('/test/<uuid:userid>/<uuid:jobid>', methods=['POST'])
def test_model(userid, jobid):
    """Tests the selected model against the provided data"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    reply = predict.predict(
        [float(x) for x in request.form['data'].split(',')],
        folder + '/pipeline'
    )

    reply['target'] = label_column

    return jsonify(reply)

@APP.route('/train/<uuid:userid>/<uuid:jobid>', methods=['POST'])
def find_best_model(userid, jobid):
    """Finds the best model for the selected parameters/data"""

    task = queue_training.s(userid.urn[9:], jobid.urn[9:], request.form.to_dict()).apply_async()
    return jsonify({
        "id": task.id,
        "href": url_for('task_status', task_id=task.id)
    }), 202

@APP.route('/status/<task_id>')
def task_status(task_id):
    return jsonify(get_task_status(task_id))

@APP.route('/results/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def get_results(userid, jobid):
    """Retrieve the training results"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]
    metadata = None

    if not os.path.exists(folder + '/report.csv'):
        abort(404)
        return

    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    return jsonify({
        'results': json.loads(pd.read_csv(folder + '/report.csv').to_json(orient='records')),
        'metadata': metadata
    })

@APP.route('/upload/<uuid:userid>/<uuid:jobid>', methods=['POST'])
def upload_files(userid, jobid):
    """Upload files to the server"""

    if 'train' not in request.files or 'test' not in request.files:
        return jsonify({'error': 'Missing files'})

    train = request.files['train']
    test = request.files['test']

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder):
        os.makedirs(folder)

    if train and test:
        train.save(folder + '/train.csv')
        test.save(folder + '/test.csv')

        label = open(folder + '/label.txt', 'w')
        label.write(request.form['label_column'])
        label.close()

        return jsonify({'success': 'true'})

    return jsonify({'error': 'unknown'})

@APP.route('/list-pending/<uuid:userid>', methods=['GET'])
def list_pending(userid):
    """Get all pending tasks for a given user ID"""

    active = []
    scheduled = []
    i = CELERY.control.inspect()

    for worker in list(i.scheduled().values()):
        for task in worker:
            if str(userid) in task['request']['args']:
                try:
                    args = ast.literal_eval(task['request']['args'])
                except:
                    continue

                scheduled.append({
                    'eta': task['eta'],
                    'jobid': args[1],
                    'parameters': args[2],
                    'state': 'PENDING'
                })

    for worker in list(i.active().values()):
        for task in worker:
            if '.queue_training' in task['type'] and str(userid) in task['args']:
                status = get_task_status(task['id'])
                active.append(status)

    return jsonify({
        'active': active,
        'scheduled': scheduled
    })

@APP.route('/list-jobs/<uuid:userid>', methods=['GET'])
def list_jobs(userid):
    """Get all the jobs for a given user ID"""

    folder = 'data/' + userid.urn[9:]

    if not os.path.exists(folder):
        abort(404)
        return

    jobs = []
    for job in os.listdir(folder):
        if not os.path.isdir(folder + '/' + job):
            continue

        has_results = os.path.exists(folder + '/' + job + '/report.csv')
        label = open(folder + '/' + job + '/label.txt', 'r')
        label_column = label.read()
        label.close()

        if os.path.exists(folder + '/' + job + '/metadata.json'):
            with open(folder + '/' + job + '/metadata.json') as json_file:
                metadata = json.load(json_file)
        else:
            metadata = {}

        jobs.append({
            'id': job,
            'label': label_column,
            'results': has_results,
            'metadata': metadata
        })

    return jsonify(jobs)

@APP.route('/export/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def export_results(userid, jobid):
    """Export the results CSV"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/report.csv'):
        abort(404)
        return

    return send_file(folder + '/report.csv', as_attachment=True)

@APP.route('/export-pmml/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def export_pmml(userid, jobid):
    """Export the selected model's PMML"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.pmml'):
        abort(404)
        return

    return send_file(folder + '/pipeline.pmml', as_attachment=True)

@APP.route('/export-pmml/<string:model>', methods=['GET'])
def export_published_pmml(model):
    """Export the published model's PMML"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(404)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(404)
        return

    if not os.path.exists(published[model]['path'] + '.pmml'):
        abort(404)
        return

    return send_file(published[model]['path'] + '.pmml', as_attachment=True)

@APP.route('/export-model/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def export_model(userid, jobid):
    """Export the selected model"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.joblib'):
        abort(404)
        return

    return send_file(folder + '/pipeline.joblib', as_attachment=True)

@APP.route('/export-model/<string:model>', methods=['GET'])
def export_published_model(model):
    """Export the published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(404)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(404)
        return

    if not os.path.exists(published[model]['path'] + '.joblib'):
        abort(404)
        return

    return send_file(published[model]['path'] + '.joblib', as_attachment=True)

@APP.route('/<path:path>')
def get_static_file(path):
    """Retrieve static files from the UI path"""

    if not os.path.isfile(os.path.join('static', path)):
        path = os.path.join(path, 'index.html')

    return send_from_directory('static', path)

def get_task_status(task_id):
    """Get's a given's task and returns a summary in JSON format"""

    task = queue_training.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }

    return response

if __name__ == "__main__":
    APP.run()
