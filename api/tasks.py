"""
API methods for getting data from Celery tasks
"""

import json

from flask import abort, g, jsonify

from worker import CELERY, get_task_status, revoke_task

def status(task_id):
    """Get a jobs status"""
    return jsonify(get_task_status(task_id.urn[9:]))

def pending():
    """Get all pending tasks for a given user ID"""

    if g.uid is None:
        abort(401)
        return

    active = []
    scheduled = []
    i = CELERY.control.inspect()
    scheduled_tasks = i.scheduled()
    scheduled_tasks = list(scheduled_tasks.values()) if scheduled_tasks else []
    for worker in scheduled_tasks:
        for task in worker:
            if g.uid in task['request']['args']:
                job_folder = 'data/users/' + g.uid + '/jobs/' + task['request']['args'][1]

                with open(job_folder + '/metadata.json') as metafile:
                    metadata = json.load(metafile)

                scheduled.append({
                    'id': task['request']['id'],
                    'eta': task['eta'],
                    'datasetid': metadata['datasetid'],
                    'jobid': task['request']['args'][1],
                    'label': task['request']['args'][2],
                    'parameters': task['request']['args'][3],
                    'state': 'PENDING'
                })

    active_tasks = i.active()
    active_tasks = list(active_tasks.values()) if active_tasks else []
    reserved_tasks = i.reserved()
    reserved_tasks = list(reserved_tasks.values()) if reserved_tasks else []
    for worker in active_tasks + reserved_tasks:
        for task in worker:
            if '.queue_training' in task['type'] and g.uid in task['args']:
                job_folder = 'data/users/' + g.uid + '/jobs/' + task['args'][1]

                with open(job_folder + '/metadata.json') as metafile:
                    metadata = json.load(metafile)

                task_status = get_task_status(task['id'])
                task_status.update({
                    'id': task['id'],
                    'datasetid': metadata['datasetid'],
                    'jobid': task['args'][1],
                    'label': task['args'][2],
                    'parameters': task['args'][3],
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
