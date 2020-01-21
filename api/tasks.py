"""
API methods for getting data from Celery tasks
"""

import ast

from flask import jsonify

from worker import CELERY, get_task_status, revoke_task

def status(task_id):
    """Get a jobs status"""
    return jsonify(get_task_status(task_id.urn[9:]))

def pending(userid):
    """Get all pending tasks for a given user ID"""

    active = []
    scheduled = []
    i = CELERY.control.inspect()
    scheduled_tasks = i.scheduled()
    scheduled_tasks = list(scheduled_tasks.values()) if scheduled_tasks else []
    for worker in scheduled_tasks:
        for task in worker:
            if userid in task['request']['args']:
                try:
                    args = ast.literal_eval(task['request']['args'])
                except ValueError:
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
            if '.queue_training' in task['type'] and userid in task['args']:
                try:
                    args = ast.literal_eval(task['args'])
                except ValueError:
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
