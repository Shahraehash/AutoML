import os
import json

from celery import Celery

from api import api

CELERY = Celery(__name__, backend='rpc://', broker='pyamqp://guest@localhost//')
CELERY.conf.update(task_track_started=True)

@CELERY.task(bind=True)
def queue_training(self, userid, jobid, label_column, parameters):
    folder = 'data/' + userid + '/' + jobid
    labels = ['No ' + label_column, label_column]

    os.environ['IGNORE_ESTIMATOR'] = parameters['ignore_estimator']
    os.environ['IGNORE_FEATURE_SELECTOR'] = parameters['ignore_feature_selector']
    os.environ['IGNORE_SCALER'] = parameters['ignore_scaler']
    os.environ['IGNORE_SEARCHER'] = parameters['ignore_searcher']
    os.environ['IGNORE_SCORER'] = parameters['ignore_scorer']
    if 'ignore_shuffle' in parameters:
        os.environ['IGNORE_SHUFFLE'] = parameters['ignore_shuffle']

    metadata = {}
    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    metadata['parameters'] = parameters

    with open(folder + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

    api.find_best_model(
        folder + '/train.csv',
        folder + '/test.csv',
        labels,
        label_column,
        folder,
        lambda x, y: self.update_state(state='PROGRESS', meta={'current': x, 'total': y})
    )
    return {}

def get_task_status(task_id):
    """Get's a given's task and returns a summary in JSON format"""

    task = queue_training.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }

    return response
