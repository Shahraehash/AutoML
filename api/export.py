"""
Exports data
"""

import os
import json

from flask import abort, send_file

PUBLISHED_MODELS = 'data/published-models.json'

def results(userid, jobid):
    """Export the results CSV"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/report.csv'):
        abort(400)
        return

    return send_file(folder + '/report.csv', as_attachment=True)

def pmml(userid, jobid):
    """Export the selected model's PMML"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.pmml'):
        abort(400)
        return

    return send_file(folder + '/pipeline.pmml', as_attachment=True)

def published_pmml(model):
    """Export the published model's PMML"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(400)
        return

    if not os.path.exists(published[model]['path'] + '.pmml'):
        abort(400)
        return

    return send_file(published[model]['path'] + '.pmml', as_attachment=True)

def model(userid, jobid):
    """Export the selected model"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder + '/pipeline.joblib'):
        abort(400)
        return

    return send_file(folder + '/pipeline.joblib', as_attachment=True)

def published_model(model):
    """Export the published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if model not in published:
        abort(400)
        return

    if not os.path.exists(published[model]['path'] + '.joblib'):
        abort(400)
        return

    return send_file(published[model]['path'] + '.joblib', as_attachment=True)
