"""
Upload dataset
"""

import os

from flask import jsonify, request

def upload(userid, jobid):
    """Upload files to the server"""

    if 'train' not in request.files or 'test' not in request.files:
        return jsonify({'error': 'Missing files'})

    train = request.files['train']
    test = request.files['test']

    folder = 'data/' + userid.urn[9:] + '/' + jobid.urn[9:]

    if not os.path.exists(folder):
        os.makedirs(folder)

    if train and test:
        train.save(folder + '/train.csv')
        test.save(folder + '/test.csv')

        label = open(folder + '/label.txt', 'w')
        label.write(request.form['label_column'])
        label.close()

        return jsonify({'success': 'true'})

    return jsonify({'error': 'unknown'})

