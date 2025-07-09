"""
Search for the best model for a given dataset
"""

import ast
import os
import json
import re
import time
import uuid
import zipfile
import tarfile
import tempfile
from io import BytesIO
from shutil import copyfile, rmtree

from flask import Response, abort, g, jsonify, request, send_file, url_for
import pandas as pd

from ml.unified_classifier_manager import UnifiedClassifierManager
from ml.job_result_accessor import JobResultAccessor, ResultNotFoundError, ModelNotFoundError
from worker import queue_training

def get():
    """Get all the jobs for a given user ID"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs'

    if not os.path.exists(folder):
        abort(400)
        return

    jobs = []
    for job in os.listdir(folder):
        if not os.path.isdir(folder + '/' + job) or\
            not os.path.exists(folder + '/' + job + '/metadata.json'):
            continue

        with open(folder + '/' + job + '/metadata.json') as metafile:
            metadata = json.load(metafile)

        jobs.append({
            'date': time.strftime(
                '%Y-%m-%dT%H:%M:%SZ',
                time.gmtime(max(
                    os.path.getmtime(root) for root, _, _ in os.walk(folder + '/' + job)
                ))
            ),
            'id': job,
            'hasResults': os.path.exists(folder + '/' + job + '/report.csv'),
            'metadata': metadata
        })

    return jsonify(jobs)

def add():
    """Creates a new job"""

    if g.uid is None:
        abort(401)
        return

    try:
        datasetid = request.get_json()['datasetid']
    except KeyError:
        abort(400)
        return

    jobid = uuid.uuid4().urn[9:]

    folder = 'data/users/' + g.uid + '/jobs/' + jobid

    if not os.path.exists(folder + '/models'):
        os.makedirs(folder + '/models')

    metadata = {}
    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    metadata['datasetid'] = datasetid

    with open(folder + '/metadata.json', 'w') as metafile:
        json.dump(metadata, metafile)

    return jsonify({'id': jobid})

def delete(jobid):
    """Deletes a previous job"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder):
        abort(400)
        return

    # Clean up class results if they exist using JobResultAccessor
    try:
        accessor = JobResultAccessor(folder)
        accessor.cleanup_results()
    except Exception as e:
        print(f"Error cleaning up class results during job deletion: {e}")

    rmtree(folder)

    return jsonify({'success': True})

def train(jobid):
    """Finds the best model for the selected parameters/data"""

    if g.uid is None:
        abort(401)
        return

    parameters = request.form.to_dict()

    pipelines = UnifiedClassifierManager.list_pipeline_configurations(parameters)

    job_folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    task = queue_training.s(
        g.uid, jobid.urn[9:], dataset_metadata['label'], parameters
    ).apply_async()

    return jsonify({
        "id": task.id,
        "href": url_for('status', task_id=task.id),
        "pipelines": pipelines
    }), 202

def result(jobid):
    """Retrieve the training results"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    try:
        accessor = JobResultAccessor(folder)
        return jsonify(accessor.get_main_results())
    except ResultNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def get_pipelines(jobid):
    """Returns the pipelines for a job"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/' + '/metadata.json'):
        abort(400)
        return

    with open(folder + '/' + '/metadata.json') as json_file:
        metadata = json.load(json_file)

    return jsonify(UnifiedClassifierManager.list_pipeline_configurations(metadata['parameters']))

def refit(jobid, threshold=.5):
    """Create a static copy of the selected model"""

    if g.uid is None:
        abort(401)
        return

    job_folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    try:
        # Create UnifiedClassifierManager and load models from archives
        manager = UnifiedClassifierManager(job_folder, metadata.get('parameters', {}))
        manager.load_models_from_archives()
        
        # Create static model using the new class-based approach
        generalization_result = manager.create_static_model(
            request.form['key'],
            ast.literal_eval(request.form['parameters']),
            ast.literal_eval(request.form['features']),
            dataset_folder,
            dataset_metadata['label'],
            threshold
        )
        
        return jsonify({'generalization': generalization_result})
        
    except Exception as e:
        return jsonify({'error': f'Failed to create static model: {str(e)}'}), 500

def tandem(jobid):
    """Create a static copy of the two selected models to be used in tandem"""

    if g.uid is None:
        abort(401)
        return

    job_folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    try:
        # Create UnifiedClassifierManager and load models from archives
        manager = UnifiedClassifierManager(job_folder, metadata.get('parameters', {}))
        manager.load_models_from_archives()
        
        # Create NPV model
        npv_generalization_result = manager.create_static_model(
            request.form['npv_key'],
            ast.literal_eval(request.form['npv_parameters']),
            ast.literal_eval(request.form['npv_features']),
            dataset_folder,
            dataset_metadata['label']
        )

        # Save NPV features and copy files
        with open(job_folder + '/tandem_npv_features.json', 'w') as npv_features:
            json.dump(ast.literal_eval(request.form['npv_features']), npv_features)

        copyfile(job_folder + '/pipeline.joblib', job_folder + '/tandem_npv.joblib')
        if os.path.exists(job_folder + '/pipeline.pmml'):
            copyfile(job_folder + '/pipeline.pmml', job_folder + '/tandem_npv.pmml')
        if os.path.exists(job_folder + '/pipeline.json'):
            copyfile(job_folder + '/pipeline.json', job_folder + '/tandem_npv.json')

        # Create PPV model
        ppv_generalization_result = manager.create_static_model(
            request.form['ppv_key'],
            ast.literal_eval(request.form['ppv_parameters']),
            ast.literal_eval(request.form['ppv_features']),
            dataset_folder,
            dataset_metadata['label']
        )

        # Save PPV features and copy files
        with open(job_folder + '/tandem_ppv_features.json', 'w') as ppv_features:
            json.dump(ast.literal_eval(request.form['ppv_features']), ppv_features)

        copyfile(job_folder + '/pipeline.joblib', job_folder + '/tandem_ppv.joblib')
        if os.path.exists(job_folder + '/pipeline.pmml'):
            copyfile(job_folder + '/pipeline.pmml', job_folder + '/tandem_ppv.pmml')
        if os.path.exists(job_folder + '/pipeline.json'):
            copyfile(job_folder + '/pipeline.json', job_folder + '/tandem_ppv.json')

        return jsonify({
            'npv_generalization': npv_generalization_result,
            'ppv_generalization_result': ppv_generalization_result
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to create tandem models: {str(e)}'}), 500

def ensemble(jobid):
    """Create a static copy of the n selected models to be used in ensemble"""

    if g.uid is None:
        abort(401)
        return

    job_folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(job_folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    try:
        # Create UnifiedClassifierManager and load models from archives
        manager = UnifiedClassifierManager(job_folder, metadata.get('parameters', {}))
        manager.load_models_from_archives()
        
        total_models = int(request.form['total_models'])
        
        # Create each ensemble model
        for x in range(total_models):
            # Create static model for this ensemble member
            manager.create_static_model(
                request.form['model' + str(x) + '_key'],
                ast.literal_eval(request.form['model' + str(x) + '_parameters']),
                ast.literal_eval(request.form['model' + str(x) + '_features']),
                dataset_folder,
                dataset_metadata['label']
            )

            # Save features for this model
            with open(job_folder + '/ensemble' + str(x) + '_features.json', 'w') as model_features:
                json.dump(ast.literal_eval(request.form['model' + str(x) + '_features']), model_features)

            # Copy the created model files
            copyfile(job_folder + '/pipeline.joblib', job_folder + '/ensemble' + str(x) + '.joblib')
            if os.path.exists(job_folder + '/pipeline.json'):
                copyfile(job_folder + '/pipeline.json', job_folder + '/ensemble' + str(x) + '.json')

        # Generate ensemble generalization results using the new class-based approach
        reply = manager.make_ensemble_generalization(total_models, dataset_folder, dataset_metadata['label'])
        reply['total_models'] = total_models

        # Save ensemble details
        with open(job_folder + '/ensemble.json', 'w') as model_details:
            json.dump(reply, model_details)

        return jsonify(reply)
        
    except Exception as e:
        return jsonify({'error': f'Failed to create ensemble models: {str(e)}'}), 500

def test(jobid):
    """Tests the selected model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    payload = json.loads(request.data)

    # Create UnifiedClassifierManager and load static model
    manager = UnifiedClassifierManager(folder, metadata.get('parameters', {}))
    
    # Load the static pipeline model
    success = manager.load_static_model(folder + '/pipeline.joblib')
    
    if success:
        # Use the new static model prediction method
        reply = manager.make_predictions(
            payload['data'],
            'pipeline',
            payload['threshold']
        )
        reply['target'] = metadata['label']
        return jsonify(reply)
    else:
        raise Exception("Failed to load static model")
        


def test_tandem(jobid):
    """Tests the selected tandem model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    payload = json.loads(request.data)

    # Create UnifiedClassifierManager and load tandem models
    manager = UnifiedClassifierManager(folder, metadata.get('parameters', {}))
    
    # Load the tandem models
    success = manager.load_tandem_models(folder)
    
    if success:
        # Use the new tandem model prediction method
        reply = manager.make_tandem_predictions(
            payload['data'],
            payload['features']
        )
        reply['target'] = metadata['label']
        return jsonify(reply)
    else:
        raise Exception("Failed to load tandem models")

def test_ensemble(jobid):
    """Tests the selected ensemble model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    payload = json.loads(request.data)

    with open(folder + '/ensemble.json') as details:
        model_details = json.load(details)

    # Create UnifiedClassifierManager and load ensemble models
    manager = UnifiedClassifierManager(folder, metadata.get('parameters', {}))
    
    # Load the ensemble models
    success = manager.load_ensemble_models(folder, model_details['total_models'])
    
    if success:
        # Use the new ensemble model prediction method
        reply = manager.make_ensemble_predictions_direct(
            payload['data'],
            payload['features'],
            payload['vote_type']
        )
        reply['target'] = metadata['label']
        return jsonify(reply)
    else:
        raise Exception("Failed to load ensemble models")

def generalize(jobid):
    """Returns generalization results for a given dataset"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    dataset_folder = 'data/users/' + g.uid + '/datasets/' + metadata['datasetid']

    with open(dataset_folder + '/metadata.json') as metafile:
        dataset_metadata = json.load(metafile)

    payload = json.loads(request.data)

    if 'data' not in payload['data']:
        test_data = pd.read_csv(dataset_folder + '/test.csv')
        payload['data']['data'] = test_data
        payload['data']['columns'] = test_data.columns


    # Create UnifiedClassifierManager and load static model
    manager = UnifiedClassifierManager(folder, metadata.get('parameters', {}))
    
    # Load the static pipeline model
    success = manager.load_static_model(folder + '/pipeline.joblib')
    
    if success:
        # Use the new static model evaluation methods
        return jsonify({
            'generalization': manager.get_static_generalization_metrics(payload['data'], 'pipeline', dataset_metadata['label'], payload['threshold']),
            'reliability': manager.get_static_reliability_metrics(payload['data'], 'pipeline', dataset_metadata['label']),
            'precision_recall': manager.get_static_precision_metrics(payload['data'], 'pipeline', dataset_metadata['label']),
            'roc_auc': manager.get_static_roc_metrics(payload['data'], 'pipeline', dataset_metadata['label'])
        })
    else:
        raise Exception("Failed to load static model")
        


def get_class_specific_results(jobid, class_index):
    """Returns class-specific results for multiclass models"""
    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    model_key = request.args.get('model_key')
    
    try:
        accessor = JobResultAccessor(folder)
        return jsonify(accessor.get_class_results(class_index, model_key))
    except (ResultNotFoundError, ModelNotFoundError) as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def export(jobid, class_index=None):
    """Export the results CSV - subset by class if specified"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    # Get class_index from query parameters if not passed as argument
    if class_index is None:
        class_index = request.args.get('class_index')
    
    try:
        accessor = JobResultAccessor(folder)
        df_filtered, filename = accessor.get_csv_export_data(class_index)
        
        return Response(
            df_filtered.to_csv(index=False),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment;filename={filename}'}
        )
    except ResultNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def export_performance(jobid):
    """Export the performance result CSV"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/performance_report.csv'):
        abort(400)
        return

    return send_file(folder + '/performance_report.csv', as_attachment=True, cache_timeout=-1)

def export_pmml(jobid, class_index=None):
    """Export the selected model's PMML - main model or OvR model"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if class_index is not None:
        # Export OvR model PMML for specific class (if it exists)
        pmml_path = folder + f'/models/ovr_class_{class_index}.pmml'
        filename = f'ovr_class_{class_index}.pmml'
    else:
        # Original main model PMML
        pmml_path = folder + '/pipeline.pmml'
        filename = 'pipeline.pmml'

    if not os.path.exists(pmml_path):
        abort(400)
        return

    return send_file(pmml_path, as_attachment=True, cache_timeout=-1, download_name=filename)

def export_model(jobid, class_index=None, model_key=None):
    """Export the selected model - main model or OvR model from compressed archives"""
    if g.uid is None:
        abort(401)
        return

    threshold = float(request.args.get('threshold', .5))
    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    # Get model_key from request if not provided as parameter
    if model_key is None:
        model_key = request.args.get('model_key')
    
    try:
        accessor = JobResultAccessor(folder)
        memory_file = accessor.create_model_export_zip(model_key, class_index, threshold)
        
        if class_index is not None:
            zip_filename = f'ovr_class_{class_index}_model.zip'
        else:
            zip_filename = 'model.zip'
        
        return send_file(memory_file, attachment_filename=zip_filename, as_attachment=True, cache_timeout=-1)
    
    except ModelNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def star_models(jobid):
    """Marks the selected models as starred"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    try:
        accessor = JobResultAccessor(folder)
        accessor.update_starred_models(models_to_add=request.get_json()['models'])
        return jsonify({'success': True})
    except ResultNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def un_star_models(jobid):
    """Removes the selected models as starred models"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    try:
        accessor = JobResultAccessor(folder)
        accessor.update_starred_models(models_to_remove=request.get_json()['models'])
        return jsonify({'success': True})
    except ResultNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def get_starred(jobid):
    """Get the starred models for a job"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    try:
        accessor = JobResultAccessor(folder)
        return jsonify(accessor.get_starred_models())
    except ResultNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def list_available_models(jobid, class_index=None):
    """List models available in archives"""
    
    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    try:
        accessor = JobResultAccessor(folder)
        return jsonify(accessor.list_available_models(class_index))
    except ResultNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code

def get_available_class_models(jobid):
    """Get list of models that have class-specific results"""
    
    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    try:
        accessor = JobResultAccessor(folder)
        return jsonify(accessor.get_available_class_models())
    except ResultNotFoundError as e:
        return jsonify({'error': str(e), 'code': e.code}), e.status_code
