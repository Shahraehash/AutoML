"""
Create a new job from a prior dataset
"""

import os
from shutil import copyfile

from flask import abort, jsonify

def clone(userid, jobid, newjobid):
    """Copies the data source to a new job ID"""

    src_folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]
    dest_folder = 'data/' + userid.urn[9:] + '/' + newjobid.urn[9:]

    if not os.path.exists(src_folder) or\
        not os.path.exists(src_folder + '/train.csv') or\
        not os.path.exists(src_folder + '/test.csv') or\
        not os.path.exists(src_folder + '/label.txt'):
        abort(400)
        return

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    copyfile(src_folder + '/train.csv', dest_folder + '/train.csv')
    copyfile(src_folder + '/test.csv', dest_folder + '/test.csv')
    copyfile(src_folder + '/label.txt', dest_folder + '/label.txt')

    return jsonify({'success': True})
