"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

import os

from firebase_admin import auth, credentials, initialize_app
from flask import Flask, g, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

import api.datasets as datasets
import api.jobs as jobs
import api.published as published
import api.tasks as tasks

load_dotenv()

APP = Flask(__name__, static_url_path='')
APP.config['JSON_SORT_KEYS'] = False
CORS(APP)

if os.path.exists('serviceAccountKey.json'):
    initialize_app(
        credentials.Certificate('serviceAccountKey.json')
    )

@APP.route('/')
def load_ui():
    """Loads `index.html` for the root path"""

    return send_from_directory('static', 'index.html')

@APP.errorhandler(404)
def page_not_found(_):
    """Redirect all invalid pages back to the root index"""

    return load_ui()

@APP.before_request
def parse_auth():
    """Handle authentication headers/query parameters for the incoming request"""

    bearer = request.headers.get('Authorization')
    current_user = request.args.get('currentUser')

    if bearer is not None:
        g.uid = auth.verify_id_token(bearer.split()[1])['uid']
        return

    if current_user is not None:
        g.uid = auth.verify_id_token(current_user)['uid']
        return

    local_user = request.headers.get('LocalUserID', request.args.get('localUser'))
    if os.getenv('NO_NETWORK_ALLOWED') == 'true' and local_user:
        g.uid = local_user
        return

    g.uid = None
    return

# Datasets
APP.add_url_rule('/datasets', 'datasets-get', datasets.get)
APP.add_url_rule('/datasets', 'datasets-add', datasets.add, methods=['POST'])
APP.add_url_rule('/datasets/<uuid:datasetid>', 'datasets-delete', datasets.delete, methods=['DELETE'])
APP.add_url_rule('/datasets/<uuid:datasetid>/describe', 'datasets-describe', datasets.describe)

# Jobs
APP.add_url_rule('/jobs', 'jobs-get', jobs.get)
APP.add_url_rule('/jobs', 'jobs-add', jobs.add, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>', 'jobs-delete', jobs.delete, methods=['DELETE'])
APP.add_url_rule('/jobs/<uuid:jobid>/train', 'jobs-train', jobs.train, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/result', 'jobs-result', jobs.result)
APP.add_url_rule('/jobs/<uuid:jobid>/refit', 'jobs-refit', jobs.refit, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/test', 'jobs-test', jobs.test, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/tandem', 'jobs-tandem', jobs.tandem, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/ensemble', 'jobs-ensemble', jobs.ensemble, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/generalize', 'jobs-generalize', jobs.generalize, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/test-tandem', 'jobs-test-tandem', jobs.test_tandem, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/test-ensemble', 'jobs-test-ensemble', jobs.test_ensemble, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/pipelines', 'jobs-pipelines', jobs.get_pipelines)
APP.add_url_rule('/jobs/<uuid:jobid>/export', 'jobs-export', jobs.export)
APP.add_url_rule('/jobs/<uuid:jobid>/export-pmml', 'jobs-export-pmml', jobs.export_pmml)
APP.add_url_rule('/jobs/<uuid:jobid>/export-model', 'jobs-export-model', jobs.export_model)
APP.add_url_rule('/jobs/<uuid:jobid>/star-models', 'jobs-get-starred', jobs.get_starred, methods=['GET'])
APP.add_url_rule('/jobs/<uuid:jobid>/star-models', 'jobs-star-models', jobs.star_models, methods=['POST'])
APP.add_url_rule('/jobs/<uuid:jobid>/un-star-models', 'jobs-un-star-models', jobs.un_star_models, methods=['POST'])

# Tasks
APP.add_url_rule('/tasks', 'pending', tasks.pending)
APP.add_url_rule('/tasks/<uuid:task_id>', 'status', tasks.status)
APP.add_url_rule('/tasks/<uuid:task_id>', 'cancel', tasks.cancel, methods=['DELETE'])

# Published Models
APP.add_url_rule('/published', 'published-get', published.get)
APP.add_url_rule('/published/<string:name>', 'published-add', published.add, methods=['POST'])
APP.add_url_rule('/published/<string:name>', 'published-delete', published.delete, methods=['DELETE'])
APP.add_url_rule('/published/<string:name>/rename', 'published-rename', published.rename, methods=['POST'])
APP.add_url_rule('/published/<string:name>/test', 'published-test', published.test, methods=['POST'])
APP.add_url_rule('/published/<string:name>/export-model', 'published-export-model', published.export_model)
APP.add_url_rule('/published/<string:name>/export-pmml', 'published-export-pmml', published.export_pmml)
APP.add_url_rule('/published/<string:name>/features', 'published-features', published.features)

if __name__ == "__main__":
    APP.run()
