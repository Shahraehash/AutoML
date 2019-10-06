"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

import os

import pandas as pd
from flask import abort, Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from api import api

APP = Flask(__name__, static_url_path='')
CORS(APP)

@APP.route('/')
def load_ui():
    return send_from_directory('static', 'index.html')

@APP.route('/train', methods=['POST'])
def run():
    label = open('data/label.txt', 'r')
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

    api.find_best_model('data/train.csv', 'data/test.csv', labels, label_column)
    return jsonify({'success': True})

@APP.route('/results', methods=['GET'])
def get_results():
    """Retrieve the training results"""

    if not os.path.exists('report.csv'):
        abort(404)
        return

    return pd.read_csv('report.csv').to_json(orient='records')

@APP.route('/export', methods=['GET'])
def export_results():
    """Export the results CSV"""

    if not os.path.exists('report.csv'):
        abort(404)
        return

    return send_file('report.csv', as_attachment=True)

@APP.route('/upload', methods=['POST'])
def upload_files():
    """Upload files to the server"""

    if 'train' not in request.files or 'test' not in request.files:
        return jsonify({'error': 'Missing files'})

    train = request.files['train']
    test = request.files['test']

    if train and test:
        train.save('data/train.csv')
        test.save('data/test.csv')

        label = open('data/label.txt', 'w')
        label.write(request.form['label_column'])
        label.close()

        return jsonify({'success': 'true'})

    return jsonify({'error': 'unknown'})

@APP.route('/<path:path>')
def get_static_file(path):
    """Retrieve static files from the UI path"""

    if not os.path.isfile(os.path.join('static', path)):
        path = os.path.join(path, 'index.html')

    return send_from_directory('static', path)

if __name__ == "__main__":
    APP.run()
