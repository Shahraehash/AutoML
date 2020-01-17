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

APP.add_url_rule('/user/<uuid:userid>/datasets', 'upload', upload, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/datasets/<uuid:datasetid>/describe', 'describe', describe_data)
APP.add_url_rule('/user/<uuid:userid>/jobs', 'create-job', jobs.create, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/train', 'train', jobs.train, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/result', 'results', results)
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/refit', 'refit', create, methods=['POST'])
APP.add_url_rule('/user/<uuid:userid>/jobs/<uuid:jobid>/test', 'test-model', test.test_model, methods=['POST'])


APP.add_url_rule('/unpublish/<string:model>', 'unpublish', unpublish, methods=['DELETE'])
APP.add_url_rule('/delete/<uuid:userid>/<uuid:jobid>', 'delete', delete, methods=['DELETE'])
APP.add_url_rule('/features/<string:model>', 'features', features, methods=['GET'])
APP.add_url_rule('/test/<string:model>', 'test-published', test.test_published_model, methods=['POST'])
APP.add_url_rule('/pipelines/<uuid:userid>/<uuid:jobid>', 'pipelines', jobs.get_pipelines, methods=['GET'])
APP.add_url_rule('/status/<task_id>', 'status', jobs.status)
APP.add_url_rule('/list-pending/<uuid:userid>', 'pending', jobs.pending, methods=['GET'])
APP.add_url_rule('/cancel/<uuid:task_id>', 'cancel', jobs.cancel, methods=['DELETE'])
APP.add_url_rule('/list-jobs/<uuid:userid>', 'list-jobs', list_jobs, methods=['GET'])
APP.add_url_rule('/list-published/<uuid:userid>', 'list-published', list_published, methods=['GET'])
APP.add_url_rule('/export/<uuid:userid>/<uuid:jobid>', 'export', export.results, methods=['GET'])
APP.add_url_rule('/export-pmml/<uuid:userid>/<uuid:jobid>', 'export-pmml', export.pmml, methods=['GET'])
APP.add_url_rule('/export-pmml/<string:model>', 'export-published-pmml', export.published_pmml, methods=['GET'])
APP.add_url_rule('/export-model/<uuid:userid>/<uuid:jobid>', 'export-model', export.model, methods=['GET'])
APP.add_url_rule('/export-model/<string:model>', 'export-published-model', export.published_model, methods=['GET'])

if __name__ == "__main__":
    APP.run()
