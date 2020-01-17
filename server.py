"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

from flask import Flask, send_from_directory
from flask_cors import CORS

import api.datasets as datasets
import api.jobs as jobs
import api.published as published
import api.tasks as tasks

APP = Flask(__name__, static_url_path='')
APP.config['JSON_SORT_KEYS'] = False
CORS(APP)

@APP.route('/')
def load_ui():
    """Loads `index.html` for the root path"""

    return send_from_directory('static', 'index.html')

@APP.errorhandler(404)
def page_not_found(_):
    """Redirect all invalid pages back to the root index"""

    return load_ui()

# Datasets
APP.add_url_rule('/user/<uuid:userid>/datasets', 'datasets-get', datasets.get)
APP.add_url_rule('/user/<uuid:userid>/datasets', 'datasets-add', datasets.add, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/datasets/<uuid:datasetid>/describe', 'datasets-describe', datasets.describe)

# Jobs
APP.add_url_rule('/user/<uuid:userid>/jobs', 'jobs-add', jobs.create, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>', 'jobs-delete', jobs.delete, methods=['DELETE'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/train', 'jobs-train', jobs.train, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/result', 'jobs-result', jobs.result)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/refit', 'jobs-refit', jobs.refit, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/test', 'jobs-test', jobs.test, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/pipelines', 'jobs-pipelines', jobs.get_pipelines)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/export', 'jobs-export', jobs.export)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/export-pmml', 'jobs-export-pmml', jobs.export_pmml)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/export-model', 'jobs-export-model', jobs.export_model)

# Tasks
APP.add_url_rule('/user/<uuid:userid>/tasks', 'pending', tasks.pending)
APP.add_url_rule('/tasks/<uuid:task_id>', 'status', tasks.status)
APP.add_url_rule('/tasks/<uuid:task_id>', 'cancel', tasks.cancel, methods=['DELETE'])

# Published Models
APP.add_url_rule('/user/<uuid:userid>/published', 'published-get', published.get)
APP.add_url_rule('/published/<string:name>', 'published-delete', published.delete, methods=['DELETE'])
APP.add_url_rule('/published/<string:name>/test', 'published-test', published.test, methods=['POST'])
APP.add_url_rule('/published/<string:name>/export-model', 'published-export-model', published.export_model)
APP.add_url_rule('/published/<string:name>/export-pmml', 'published-export-pmml', published.export_pmml)
APP.add_url_rule('/published/<string:name>/features', 'published-features', published.features)

if __name__ == "__main__":
    APP.run()
