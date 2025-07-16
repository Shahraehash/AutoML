"""
Celery worker for machine learning jobs
"""

import os
import json
import tempfile

from celery import Celery
from celery.signals import worker_process_init

from api import search

def configure_joblib_environment():
    """
    Configure joblib environment for Celery workers to prevent resource leaks
    """
    try:
        # Set worker name for identification
        worker_name = f"celery_worker_{os.getpid()}"
        os.environ['CELERY_WORKER_NAME'] = worker_name
        
        # Create worker-specific temporary directory
        temp_dir = os.path.join(tempfile.gettempdir(), f'joblib_{worker_name}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Configure joblib environment variables
        os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
        os.environ['JOBLIB_MULTIPROCESSING'] = '1'
        os.environ['LOKY_PICKLER'] = 'pickle'
        
        # Limit parallel processing to prevent resource conflicts
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        max_jobs = max(1, min(4, cpu_count // 2))
        os.environ['LOKY_MAX_CPU_COUNT'] = str(max_jobs)
        
        print(f"Configured joblib for worker {worker_name}: temp_dir={temp_dir}, max_jobs={max_jobs}")
        
    except Exception as e:
        print(f"Warning: Could not configure joblib environment: {e}")

# Configure joblib when worker module is imported
configure_joblib_environment()

BROKER_URL = os.getenv('BROKER_URL', 'pyamqp://guest@127.0.0.1//')
CELERY = Celery(__name__, backend='rpc://', broker=BROKER_URL)
CELERY.conf.update(
    task_track_started=True,
    worker_max_tasks_per_child=1,  # Restart worker after each task to prevent memory leaks
    worker_prefetch_multiplier=1,  # Process one task at a time
    task_acks_late=True,  # Acknowledge task only after completion
    worker_disable_rate_limits=True,  # Disable rate limits for memory-intensive tasks
)

def fix_celery_solo(userid, jobid):
    """
    Celery retries tasks due to ACK issues when running in solo mode,
    We can manually check if the task has already completed and quickly finish the re-queue.
    """

    folder = 'data/users/' + userid + '/' + jobid
    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            try:
                metadata = json.load(metafile)
            except ValueError:
                return False

        if 'date' in metadata:
            return True

    return False

@CELERY.task(bind=True)
def queue_training(self, userid, jobid, label_column, parameters):
    if fix_celery_solo(userid, jobid):
        return 0

    job_folder = 'data/users/' + userid + '/jobs/' + jobid

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + userid + '/datasets/' + metadata['datasetid']
    labels = ['No ' + label_column, label_column]

    metadata['parameters'] = parameters
    metadata['label'] = label_column

    with open(job_folder + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

    search.find_best_model(
        dataset_folder + '/train.csv',
        dataset_folder + '/test.csv',
        labels,
        label_column,
        parameters,
        job_folder,
        lambda x, y: self.update_state(state='PROGRESS', meta={'current': x, 'total': y})
    )
    return {}

def revoke_task(task_id):
    CELERY.control.revoke(task_id, terminate=True)

def get_task_status(task_id):
    """Gets a given's task and returns a summary in JSON format"""

    task = queue_training.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1
        }
    elif task.state != 'FAILURE':
        response = {'state': task.state}

        if isinstance(task.info, dict):
            response.update({
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 1),
                'status': task.info.get('status', '')
            })
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),
        }

    return response

@worker_process_init.connect
def fix_multiprocessing(**_):
    """
    This turns off daemon mode for celery processes.
    https://stackoverflow.com/questions/46443541/process-is-not-spawning-with-multiprocessing-module-in-celery
    """

    from multiprocessing import current_process
    current_process().daemon = False
    
    # Reconfigure joblib for this specific worker process
    configure_joblib_environment()

def cleanup_worker_resources():
    """
    Clean up worker-specific resources including joblib temporary directories
    """
    try:
        import shutil
        import glob
        
        worker_name = os.environ.get('CELERY_WORKER_NAME', f'celery_worker_{os.getpid()}')
        temp_dir = os.path.join(tempfile.gettempdir(), f'joblib_{worker_name}')
        
        # Clean up worker-specific temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up worker temp directory: {temp_dir}")
        
        # Clean up any remaining joblib folders for this worker
        temp_base = tempfile.gettempdir()
        patterns = [
            f'joblib_memmapping_folder_{os.getpid()}_*',
            f'joblib_{worker_name}*'
        ]
        
        for pattern in patterns:
            for folder_path in glob.glob(os.path.join(temp_base, pattern)):
                try:
                    if os.path.isdir(folder_path):
                        shutil.rmtree(folder_path, ignore_errors=True)
                except (OSError, PermissionError):
                    pass
                    
    except Exception as e:
        print(f"Warning: Error during worker resource cleanup: {e}")

# Register cleanup function to run when worker shuts down
import atexit
atexit.register(cleanup_worker_resources)
