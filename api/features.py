"""
Get features for a published model
"""

import os
import json

from flask import abort, jsonify

PUBLISHED_MODELS = 'data/published-models.json'

def features(name):
    """Returns the features for a published model"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    if name not in published:
        abort(400)
        return

    return jsonify(published[name]['features'])
