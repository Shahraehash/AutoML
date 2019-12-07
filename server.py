"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

import ast
import os
import json
from shutil import copyfile

import pandas as pd
from flask import abort, Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from api import api
from api import create_model
from api import predict

APP = Flask(__name__, static_url_path='')
CORS(APP)
PUBLISHED_MODELS = 'data/published-models.json'

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
        model_path = folder + '/' + request.form['publishName'] + '.joblib'
        copyfile(folder + '/pipeline.joblib', model_path)

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

@APP.route('/test/<uuid:userid>/<uuid:jobid>', methods=['POST'])
def test_model(userid, jobid):
    """Tests the selected model against the provided data"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    return jsonify(predict.predict(
        [float(x) for x in request.form['data'].split(',')],
        folder + '/train.csv',
        label_column,
        folder
    ))

@APP.route('/train/<uuid:userid>/<uuid:jobid>', methods=['POST'])
def find_best_model(userid, jobid):
    """Finds the best model for the selected parameters/data"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    labels = ['No ' + label_column, label_column]

    os.environ['IGNORE_ESTIMATOR'] = request.form['ignore_estimator']
    os.environ['IGNORE_FEATURE_SELECTOR'] = request.form['ignore_feature_selector']
    os.environ['IGNORE_SCALER'] = request.form['ignore_scaler']
    os.environ['IGNORE_SEARCHER'] = request.form['ignore_searcher']
    os.environ['IGNORE_SCORER'] = request.form['ignore_scorer']
    if request.form.get('ignore_shuffle'):
        os.environ['IGNORE_SHUFFLE'] = request.form.get('ignore_shuffle')

    api.find_best_model(folder + '/train.csv', folder + '/test.csv', labels, label_column, folder)
    return jsonify({'success': True})

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

@APP.route('/export-model/<uuid:userid>/<uuid:jobid>', methods=['GET'])
def export_model(userid, jobid):
    """Export the selected model"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.joblib'):
        abort(404)
        return

    return send_file(folder + '/pipeline.joblib', as_attachment=True)

@APP.route('/<path:path>')
def get_static_file(path):
    """Retrieve static files from the UI path"""

    if not os.path.isfile(os.path.join('static', path)):
        path = os.path.join(path, 'index.html')

    return send_from_directory('static', path)

if __name__ == "__main__":
    APP.run()
