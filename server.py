"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

import os
from flask import Flask, jsonify, request, send_from_directory

from api import api

APP = Flask(__name__, static_url_path='')

@APP.route('/')
def load_ui():
    return send_from_directory('static', 'index.html')

@APP.route('/train', methods=['POST'])
def run():
    body = request.get_json()
    results = api.find_best_model(body['train'], body['test'], body['labels'], body['label_column'])
    return jsonify(results)

@APP.route('/upload', methods=['POST'])
def upload_files():
    print(request.files)
    if 'train' not in request.files or 'test' not in request.files:
        return jsonify({'error': 'Missing files'})

    train = request.files['train']
    test = request.files['test']
    if train.filename != 'train.csv' or test.filename != 'test.csv':
        return jsonify({'error': 'Missing files'})

    if train and test:
        train.save('data/' + train.filename)
        test.save('data/' + test.filename)
        return jsonify({'success': 'true'})

    return jsonify({'error': 'unknown'})

@APP.route('/<path:path>')
def get_static_file(path):
    if not os.path.isfile(os.path.join('static', path)):
        path = os.path.join(path, 'index.html')

    return send_from_directory('static', path)

if __name__ == "__main__":
    APP.run()
