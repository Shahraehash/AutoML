"""
Generate descriptive statistics
"""

import os

from flask import abort

from ml.describe import describe

def describe_data(userid, jobid):
    """Generate descriptive statistics for training/testing datasets"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    label = open(folder + '/label.txt', 'r')
    label_column = label.read()
    label.close()

    return {
        'analysis': describe(folder),
        'label': label_column
    }
