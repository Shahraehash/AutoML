"""
Retrieve prior jobs
"""

import os
import json

from flask import abort, jsonify

def list_jobs(userid):
    """Get all the jobs for a given user ID"""

    folder = 'data/' + userid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    jobs = []
    for job in os.listdir(folder):
        if not os.path.isdir(folder + '/' + job) or\
            not os.path.exists(folder + '/' + job + '/train.csv') or\
            not os.path.exists(folder + '/' + job + '/test.csv') or\
            not os.path.exists(folder + '/' + job + '/label.txt'):
            continue

        has_results = os.path.exists(folder + '/' + job + '/report.csv')
        label = open(folder + '/' + job + '/label.txt', 'r')
        label_column = label.read()
        label.close()

        if os.path.exists(folder + '/' + job + '/metadata.json'):
            with open(folder + '/' + job + '/metadata.json') as json_file:
                metadata = json.load(json_file)
        else:
            metadata = {}

        jobs.append({
            'id': job,
            'label': label_column,
            'results': has_results,
            'metadata': metadata
        })

    return jsonify(jobs)
