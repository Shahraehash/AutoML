import os
import json

from celery import Celery

from api import api

CELERY = Celery(__name__, backend='rpc://', broker='pyamqp://guest@localhost//')
CELERY.conf.update(task_track_started=True)

@CELERY.task()
def queue_training(userid, jobid, label_column, parameters):
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

    api.find_best_model(folder + '/train.csv', folder + '/test.csv', labels, label_column, folder)
    return {}

