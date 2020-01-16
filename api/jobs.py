"""
Search for the best model for a given dataset
"""

import ast
import os
import json

from flask import abort, jsonify, request, url_for

from ml.list_pipelines import list_pipelines
from worker import CELERY, get_task_status, queue_training, revoke_task

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
    """Get a jobs status"""
    return jsonify(get_task_status(task_id))

def pending(userid):
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
                task_status = get_task_status(task['id'])
                task_status.update({
                    'id': task['id'],
                    'jobid': args[1],
                    'label': args[2],
                    'parameters': args[3],
                    'time': task['time_start']
                })
                active.append(task_status)
    return jsonify({
        'active': active,
        'scheduled': scheduled
    })

def cancel(task_id):
    """Cancels the provided task"""
    revoke_task(task_id)
    return jsonify({'success': True})
