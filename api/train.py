"""
Search for the best model for a given dataset
"""

import os
import json

from flask import abort, jsonify, request, url_for

from ml.list_pipelines import list_pipelines
from worker import get_task_status, queue_training

def train(userid, jobid):
    """Finds the best model for the selected parameters/data"""

    label = open('data/' + userid.urn[9:] + '/' + jobid.urn[9:] + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    parameters = request.form.to_dict()
    pipelines = list_pipelines(parameters)

    task = queue_training.s(
        userid.urn[9:], jobid.urn[9:], label_column, parameters
    ).apply_async()

    return jsonify({
        "id": task.id,
        "href": url_for('status', task_id=task.id),
        "pipelines": pipelines
    }), 202
    
def get_pipelines(userid, jobid):
    """Returns the pipelines for a job"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/' + '/metadata.json'):
        abort(400)
        return
    
    with open(folder + '/' + '/metadata.json') as json_file:
        metadata = json.load(json_file)

    return jsonify(list_pipelines(metadata['parameters']))

def status(task_id):
    return jsonify(get_task_status(task_id))
