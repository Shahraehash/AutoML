"""
AutoML

Launches the API server and allows access
using an Angular SPA.
"""

from flask import Flask, send_from_directory
from flask_cors import CORS

from api.create import create
from api.delete import delete
from api.describe import describe_data
import api.export as export
from api.features import features
import api.jobs as jobs
from api.prior import list_jobs
from api.published import list_published
from api.results import results
import api.test as test
from api.upload import upload 
from api.unpublish import unpublish

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
APP.add_url_rule('/user/<uuid:userid>/datasets', 'upload', upload, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/datasets/<uuid:datasetid>/describe', 'describe', describe_data)

# Jobs
APP.add_url_rule('/user/<uuid:userid>/jobs', 'list-jobs', list_jobs)
APP.add_url_rule('/user/<uuid:userid>/jobs', 'create-job', jobs.create, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>', 'delete', delete, methods=['DELETE'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/train', 'train', jobs.train, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/result', 'results', results)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/refit', 'refit', create, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/test', 'test-model', test.test_model, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/pipelines', 'pipelines', jobs.get_pipelines)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/export', 'export', export.results)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/export-pmml', 'export-pmml', export.pmml)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/export-model', 'export-model', export.model)

# Tasks
APP.add_url_rule('/user/<uuid:userid>/tasks', 'pending', jobs.pending)
APP.add_url_rule('/tasks/<uuid:task_id>', 'status', jobs.status)
APP.add_url_rule('/tasks/<uuid:task_id>', 'cancel', jobs.cancel, methods=['DELETE'])

# Published Models
APP.add_url_rule('/user/<uuid:userid>/published', 'list-published', list_published)
APP.add_url_rule('/published/<string:name>', 'unpublish', unpublish, methods=['DELETE'])
APP.add_url_rule('/published/<string:name>/test', 'test-published', test.test_published_model, methods=['POST'])
APP.add_url_rule('/published/<string:name>/export-model', 'export-published-model', export.published_model)
APP.add_url_rule('/published/<string:name>/export-pmml', 'export-published-pmml', export.published_pmml)
APP.add_url_rule('/published/<string:name>/features', 'features', features)

if __name__ == "__main__":
    APP.run()
