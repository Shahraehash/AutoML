"""
Handle published model requests
"""

import ast
import os
import json

from flask import abort, jsonify

PUBLISHED_MODELS = 'data/published-models.json'

def list_published(userid):
    """Get all published models for a given user ID"""

    if not os.path.exists(PUBLISHED_MODELS):
        abort(400)
        return

    with open(PUBLISHED_MODELS) as published_file:
        published = json.load(published_file)

    user = userid.urn[9:]
    published = {k:ast.literal_eval(v['features']) for (k, v) in published.items() if user in v['path']}

    return jsonify(published)
