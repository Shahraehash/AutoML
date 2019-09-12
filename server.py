"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

import os
from flask import Flask, send_from_directory

from api import api

APP = Flask(__name__, static_url_path='')

@APP.route('/')
def load_ui():
    return send_from_directory('static', 'index.html')

@APP.route('/<path:path>')
def get_static_file(path):
    if not os.path.isfile(os.path.join('static', path)):
        path = os.path.join(path, 'index.html')

    return send_from_directory('static', path)

@APP.route('/run')
def run():
    api.find_best_model('sample-data/train.csv', 'sample-data/test.csv', None, 'AKI')

if __name__ == "__main__":
    APP.run()
