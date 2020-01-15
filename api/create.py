"""
Handles creating a model from parameters
"""

import os
import ast
import json
from shutil import copyfile

from flask import abort, jsonify, request

from ml.create_model import create_model

PUBLISHED_MODELS = 'data/published-models.json'

def create(userid, jobid):
    """Create a static copy of the selected model"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    create_model(
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