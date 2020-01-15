"""
Get the run's results
"""

import os
import json

from flask import abort, jsonify
import pandas as pd

def results(userid, jobid):
    """Retrieve the training results"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]
    metadata = None

    if not os.path.exists(folder + '/report.csv'):
        abort(400)
        return

    try:
        results = json.loads(pd.read_csv(folder + '/report.csv').to_json(orient='records'))
    except:
        abort(400)

    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    return jsonify({
        'results': results,
        'metadata': metadata
    })