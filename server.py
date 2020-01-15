"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

import ast
import os
import json
from shutil import copyfile

from flask import abort, Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from api.create import create
from api.delete import delete
from api.describe import describe_data
from api.features import features
from api.results import results
import api.test as test
import api.train as train
from api.unpublish import unpublish
from worker import CELERY, get_task_status, revoke_task

PUBLISHED_MODELS = 'data/published-models.json'

APP = Flask(__name__, static_url_path='')
APP.config['JSON_SORT_KEYS'] = False
CORS(APP)

@APP.route('/')
def load_ui():
    """Loads `index.html` for the root path"""

    return send_from_directory('static', 'index.html')

APP.add_url_rule('/create/<uuid:userid>/<uuid:jobid>', 'create', create, methods=['POST'])
APP.add_url_rule('/unpublish/<string:model>', 'unpublish', unpublish, methods=['DELETE'])
APP.add_url_rule('/delete/<uuid:userid>/<uuid:jobid>', 'delete', delete, methods=['DELETE'])
APP.add_url_rule('/describe/<uuid:userid>/<uuid:jobid>', 'describe', describe_data, methods=['GET'])
APP.add_url_rule('/features/<string:model>', 'features', features, methods=['GET'])
APP.add_url_rule('/test/<string:model>', 'test_published', test.test_published_model, methods=['POST'])
APP.add_url_rule('/test/<uuid:userid>/<uuid:jobid>', 'test_model', test.test_model, methods=['POST'])
APP.add_url_rule('/train/<uuid:userid>/<uuid:jobid>', 'train', train.train, methods=['POST'])
APP.add_url_rule('/pipelines/<uuid:userid>/<uuid:jobid>', 'pipelines', train.get_pipelines, methods=['GET'])
APP.add_url_rule('/status/<task_id>', 'status', train.status)
APP.add_url_rule('/results/<uuid:userid>/<uuid:jobid>', 'results', results, methods=['GET'])

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

@APP.route('/clone/<uuid:userid>/<uuid:jobid>/<uuid:newjobid>', methods=['POST'])
def clone_job(userid, jobid, newjobid):
    """Copies the data source to a new job ID"""

    src_folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]
    dest_folder = 'data/' + userid.urn[9:] + '/' + newjobid.urn[9:]

    if not os.path.exists(src_folder) or\
        not os.path.exists(src_folder + '/train.csv') or\
        not os.path.exists(src_folder + '/test.csv') or\
        not os.path.exists(src_folder + '/label.txt'):
        abort(400)
        return

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    copyfile(src_folder + '/train.csv', dest_folder + '/train.csv')
    copyfile(src_folder + '/test.csv', dest_folder + '/test.csv')
    copyfile(src_folder + '/label.txt', dest_folder + '/label.txt')

    return jsonify({'success': True})

@APP.route('/list-pending/<uuid:userid>', methods=['GET'])
def list_pending(userid):
    """Get all pending tasks for a given user ID"""

    active = []
    scheduled = []
    i = CELERY.control.inspect()
    scheduled_tasks = i.scheduled()
    scheduled_tasks = list(scheduled_tasks.values()) if scheduled_tasks else []
    for worker in scheduled_tasks:
        for task in worker:
            if str(userid) in task['request']['args']:
                try:
                    args = ast.literal_eval(task['request']['args'])
                except:
                    continue
                scheduled.append({
                    'id': task['request']['id'],
                    'eta': task['eta'],
                    'jobid': args[1],
                    'label': args[2],
                    'parameters': args[3],
                    'state': 'PENDING'
                })

    active_tasks = i.active()
    active_tasks = list(active_tasks.values()) if active_tasks else []
    reserved_tasks = i.reserved()
    reserved_tasks = list(reserved_tasks.values()) if reserved_tasks else []
    for worker in active_tasks + reserved_tasks:
        for task in worker:
            if '.queue_training' in task['type'] and str(userid) in task['args']:
                try:
                    args = ast.literal_eval(task['args'])
                except:
                    continue
                status = get_task_status(task['id'])
                status.update({
                    'id': task['id'],
                    'jobid': args[1],
                    'label': args[2],
                    'parameters': args[3],
                    'time': task['time_start']
                })
                active.append(status)
    return jsonify({
        'active': active,
        'scheduled': scheduled
    })

@APP.route('/cancel/<uuid:task_id>', methods=['DELETE'])
def cancel_task(task_id):
    """Cancels the provided task"""
    revoke_task(task_id)
    return jsonify({'success': True})

@APP.route('/list-jobs/<uuid:userid>', methods=['GET'])
def list_jobs(userid):
    """Get all the jobs for a given user ID"""

    folder = 'data/' + userid.urn[9:]

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

@APP.route('/list-published/<uuid:userid>', methods=['GET'])
def list_published(userid):
    """Get all published models for a given user ID"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    user = userid.urn[9:]
    published = {k:ast.literal_eval(v['features']) for (k,v) in published.items() if user in v['path']}

    return jsonify(published)

@APP.route('/export/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def export_results(userid, jobid):
    """Export the results CSV"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/report.csv'):
        abort(400)
        return

    return send_file(folder + '/report.csv', as_attachment=True)

@APP.route('/export-pmml/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def export_pmml(userid, jobid):
    """Export the selected model's PMML"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.pmml'):
        abort(400)
        return

    return send_file(folder + '/pipeline.pmml', as_attachment=True)

@APP.route('/export-pmml/<string:model>', methods=['GET'])
def export_published_pmml(model):
    """Export the published model's PMML"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(400)
        return

    if not os.path.exists(published[model]['path'] + '.pmml'):
        abort(400)
        return

    return send_file(published[model]['path'] + '.pmml', as_attachment=True)

@APP.route('/export-model/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def export_model(userid, jobid):
    """Export the selected model"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.joblib'):
        abort(400)
        return

    return send_file(folder + '/pipeline.joblib', as_attachment=True)

@APP.route('/export-model/<string:model>', methods=['GET'])
def export_published_model(model):
    """Export the published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(400)
        return

    if not os.path.exists(published[model]['path'] + '.joblib'):
        abort(400)
        return

    return send_file(published[model]['path'] + '.joblib', as_attachment=True)

@APP.errorhandler(404)
def page_not_found(e):
    """Redirect all invalid pages back to the root index"""

    return load_ui()

if __name__ == "__main__":
    APP.run()
