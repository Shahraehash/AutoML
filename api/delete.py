"""
Deletes a job from storage
"""

import os
from shutil import rmtree

from flask import abort, jsonify

def delete(userid, jobid):
    """Deletes a previous job"""

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    rmtree(folder)

    return jsonify({'success': True})