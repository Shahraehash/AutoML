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


import csv
import itertools
from ml.processors.estimators import ESTIMATOR_NAMES, ESTIMATORS
from ml.processors.feature_selection import FEATURE_SELECTOR_NAMES, FEATURE_SELECTORS
from ml.processors.scalers import SCALER_NAMES, SCALERS
from ml.processors.searchers import SEARCHER_NAMES, SEARCHERS
from ml.processors.debug import Debug
from ml.utils.preprocess import preprocess
from ml.utils.utils import model_key_to_name, explode_key
from ml.utils.stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci
from ml.utils.summary import print_summary
from ml.utils.list_pipelines import list_pipelines
from ml.utils.class_results import load_class_results, get_available_models_with_class_results, cleanup_class_results

# Import classifier classes for static methods
from ml.binary_classifier import BinaryClassifier
from ml.multiclass_macro_classifier import MulticlassMacroClassifier
from ml.multiclass_ovr_classifier import OvRClassifier

from worker import queue_training

def _determine_classification_type_and_classifier(metadata, test_data=None, dataset_folder=None, label_column=None):
    """
    Helper function to determine classification type and return appropriate classifier class.
    
    Args:
        metadata (dict): Job metadata containing parameters
        test_data (pd.DataFrame, optional): Test data for class detection
        dataset_folder (str, optional): Path to dataset folder
        label_column (str, optional): Name of label column
        
    Returns:
        tuple: (n_classes, classifier_class, reoptimize_ovr)
    """
    # Try to get class count from test data if provided
    if test_data is not None and len(test_data.columns) > 0:
        if label_column and label_column in test_data.columns:
            n_classes = len(pd.Series(test_data[label_column]).unique())
        else:
            # Fallback: assume last column is the label
            n_classes = len(test_data.iloc[:, -1].unique())
    elif dataset_folder and label_column:
        # Load test data from dataset folder
        test_data_path = dataset_folder + '/test.csv'
        if os.path.exists(test_data_path):
            test_data = pd.read_csv(test_data_path)
            n_classes = len(pd.Series(test_data[label_column]).unique())
    else:
        print("ERROR: failed in _determine_classificaition_type_and_classifier")
        n_classes = 2 
    
    # Check if OvR was used (from training parameters)
    reoptimize_ovr = metadata.get('parameters', {}).get('reoptimize_ovr', 'false').lower() == 'true'
    
    # Determine appropriate classifier class
    if n_classes == 2:
        classifier_class = BinaryClassifier
    elif n_classes > 2 and reoptimize_ovr:
        classifier_class = OvRClassifier
    else:
        classifier_class = MulticlassMacroClassifier
    
    return n_classes, classifier_class, reoptimize_ovr

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

    
    try:
        class_results_dir = folder + '/class_results'
        if os.path.exists(class_results_dir):
            try:
                cleanup_class_results(class_results_dir)
            except Exception as e:
                print(f"Error cleaning up class results during job deletion: {e}")
    
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

    pipelines = list_pipelines(parameters)

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
    metadata = None

    if not os.path.exists(folder + '/report.csv'):
        abort(400)
        return
    try:
        data = json.loads(pd.read_csv(folder + '/report.csv').to_json(orient='records'))
    except ValueError:
        abort(400)

    if os.path.exists(folder + '/metadata.json'):
        with open(folder + '/metadata.json') as metafile:
            metadata = json.load(metafile)

    return jsonify({
        'results': data,
        'metadata': metadata
    })

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

    return jsonify(list_pipelines(metadata['parameters']))

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

    # Determine classification type and appropriate classifier
    n_classes, classifier_class, reoptimize_ovr = _determine_classification_type_and_classifier(
        metadata, dataset_folder=dataset_folder, label_column=dataset_metadata['label']
    )
    
    # Use appropriate static method based on classification type
    generalization_result = classifier_class.create_model(
        request.form['key'],
        ast.literal_eval(request.form['parameters']),
        ast.literal_eval(request.form['features']),
        dataset_folder,
        dataset_metadata['label'],
        job_folder,
        threshold
    )

    return jsonify({'generalization': generalization_result})

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

    # Determine classification type and appropriate classifier
    n_classes, classifier_class, reoptimize_ovr = _determine_classification_type_and_classifier(
        metadata, dataset_folder=dataset_folder, label_column=dataset_metadata['label']
    )
    
    # Use appropriate static method for NPV model
    npv_generalization_result = classifier_class.create_model(
        request.form['npv_key'],
        ast.literal_eval(request.form['npv_parameters']),
        ast.literal_eval(request.form['npv_features']),
        dataset_folder,
        dataset_metadata['label'],
        job_folder
    )

    with open(job_folder + '/tandem_npv_features.json', 'w') as npv_features:
        json.dump(ast.literal_eval(request.form['npv_features']), npv_features)

    copyfile(job_folder + '/pipeline.joblib', job_folder + '/tandem_npv.joblib')
    copyfile(job_folder + '/pipeline.pmml', job_folder + '/tandem_npv.pmml')
    copyfile(job_folder + '/pipeline.json', job_folder + '/tandem_npv.json')

    # Use appropriate static method for PPV model (same classification type)
    ppv_generalization_result = classifier_class.create_model(
        request.form['ppv_key'],
        ast.literal_eval(request.form['ppv_parameters']),
        ast.literal_eval(request.form['ppv_features']),
        dataset_folder,
        dataset_metadata['label'],
        job_folder
    )

    with open(job_folder + '/tandem_ppv_features.json', 'w') as ppv_features:
        json.dump(ast.literal_eval(request.form['ppv_features']), ppv_features)

    copyfile(job_folder + '/pipeline.joblib', job_folder + '/tandem_ppv.joblib')
    copyfile(job_folder + '/pipeline.pmml', job_folder + '/tandem_ppv.pmml')
    copyfile(job_folder + '/pipeline.json', job_folder + '/tandem_ppv.json')


    return jsonify({
        'npv_generalization': npv_generalization_result,
        'ppv_generalization_result': ppv_generalization_result
    })

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

    # Determine classification type and appropriate classifier
    n_classes, classifier_class, reoptimize_ovr = _determine_classification_type_and_classifier(
        metadata, dataset_folder=dataset_folder, label_column=dataset_metadata['label']
    )
    
    total_models = int(request.form['total_models'])
    for x in range(total_models):
        # Use appropriate static method for each ensemble model
        classifier_class.create_model(
            request.form['model' + str(x) + '_key'],
            ast.literal_eval(request.form['model' + str(x) + '_parameters']),
            ast.literal_eval(request.form['model' + str(x) + '_features']),
            dataset_folder,
            dataset_metadata['label'],
            job_folder
        )

        with open(job_folder + '/ensemble' + str(x) + '_features.json', 'w') as model_features:
            json.dump(ast.literal_eval(request.form['model' + str(x) + '_features']), model_features)

        copyfile(job_folder + '/pipeline.joblib', job_folder + '/ensemble' + str(x) + '.joblib')
        copyfile(job_folder + '/pipeline.json', job_folder + '/ensemble' + str(x) +'.json')

    # Use appropriate static method for ensemble generalization
    reply = classifier_class.generalize_ensemble(total_models, job_folder, dataset_folder, dataset_metadata['label'])
    reply['total_models'] = total_models

    with open(job_folder + '/ensemble.json', 'w') as model_details:
        json.dump(reply, model_details)

    return jsonify(reply)

def test(jobid):
    """Tests the selected model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    payload = json.loads(request.data)

    # Determine classification type and appropriate classifier
    test_data_df = pd.DataFrame(payload['data'])
    n_classes, classifier_class, reoptimize_ovr = _determine_classification_type_and_classifier(
        metadata, test_data=test_data_df
    )
    
    # Use appropriate static method based on classification type
    reply = classifier_class.predict(payload['data'], folder + '/pipeline', payload['threshold'])

    reply['target'] = metadata['label']
    return jsonify(reply)

def test_tandem(jobid):
    """Tests the selected tandem model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    with open(folder + '/tandem_npv_features.json') as feature_file:
        npv_features = json.load(feature_file)

    payload = json.loads(request.data)
    data = pd.DataFrame(payload['data'], columns=payload['features'])

    # Determine classification type and appropriate classifier
    n_classes, classifier_class, reoptimize_ovr = _determine_classification_type_and_classifier(
        metadata, test_data=data
    )

    # Use appropriate static method for NPV model based on classification type
    npv_result = classifier_class.predict(
        data[npv_features].to_numpy().tolist(),
        folder + '/tandem_npv',
        payload.get('threshold', 0.5)
    )

    npv_reply = pd.DataFrame(npv_result)

    with open(folder + '/tandem_ppv_features.json') as feature_file:
        ppv_features = json.load(feature_file)

    # Use appropriate static method for PPV model based on classification type
    ppv_result = classifier_class.predict(
        data[ppv_features].to_numpy().tolist(),
        folder + '/tandem_ppv',
        payload.get('threshold', 0.5)
    )

    ppv_reply = pd.DataFrame(ppv_result)

    ppv_reply['predicted'] = ppv_reply.apply(lambda row: row['predicted'] if row['predicted'] > 0 else -1, axis=1)
    predicted = npv_reply.apply(lambda row: ppv_reply.iloc[row.name]['predicted'] if row['predicted'] > 0 else row['predicted'], axis=1)
    npv_reply['probability'] = npv_reply.apply(lambda row: ppv_reply.iloc[row.name]['probability'] if row['predicted'] > 0 else row['probability'], axis=1)
    npv_reply['predicted'] = predicted

    return jsonify({
      'predicted': npv_reply['predicted'].to_list(),
      'probability': npv_reply['probability'].to_list(),
      'target': metadata['label']
    })

def test_ensemble(jobid):
    """Tests the selected ensemble model against the provided data"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    with open(folder + '/metadata.json') as metafile:
        metadata = json.load(metafile)

    payload = json.loads(request.data)
    data = pd.DataFrame(payload['data'], columns=payload['features'])

    with open(folder + '/ensemble.json') as details:
        model_details = json.load(details)

    # Determine classification type and appropriate classifier
    n_classes, classifier_class, reoptimize_ovr = _determine_classification_type_and_classifier(
        metadata, test_data=data
    )
    
    # Use appropriate static method based on classification type
    reply = classifier_class.predict_ensemble(
        model_details['total_models'], data, folder, payload['vote_type']
    )

    reply['target'] = metadata['label']

    return jsonify(reply)

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

    # Determine classification type and appropriate classifier
    test_data_df = pd.DataFrame(payload['data']['data'], columns=payload['data']['columns'])
    n_classes, classifier_class, reoptimize_ovr = _determine_classification_type_and_classifier(
        metadata, test_data=test_data_df, label_column=dataset_metadata['label']
    )
    
    # Use appropriate static method based on classification type
    generalization_result = classifier_class.generalize_model(
        payload['data'], dataset_metadata['label'], folder + '/pipeline', payload['threshold']
    )
    reliability_result = classifier_class.additional_reliability(
        payload['data'], dataset_metadata['label'], folder + '/pipeline'
    )
    precision_result = classifier_class.additional_precision(
        payload['data'], dataset_metadata['label'], folder + '/pipeline'
    )
    roc_result = classifier_class.additional_roc(
        payload['data'], dataset_metadata['label'], folder + '/pipeline'
    )

    return jsonify({
        'generalization': generalization_result,
        'reliability': reliability_result,
        'precision_recall': precision_result,
        'roc_auc': roc_result
    })

def get_class_specific_results(jobid, class_index):
    """Returns class-specific results for multiclass models"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    class_results_dir = folder + '/class_results'
    
    # Get model key from request parameters - if not provided, try to get the first available model
    model_key = request.args.get('model_key')

    try:
        available_models = get_available_models_with_class_results(class_results_dir)
        
        if not available_models:
            return jsonify({
                'error': 'No class-specific results available for this job.',
                'code': 'NO_CLASS_RESULTS'
            }), 400
        
        # If no model_key specified, use the first available model
        if not model_key:
            model_key = available_models[0]
            
        class_data = load_class_results(class_results_dir, model_key)
        
        if not class_data:
            return jsonify({
                'error': f'Class-specific results not found for model {model_key}.',
                'code': 'MODEL_NOT_FOUND'
            }), 404
            
        class_index = int(class_index)
        
        if class_index not in class_data['class_data']:
            return jsonify({
                'error': f'Class {class_index} not found in results.',
                'code': 'CLASS_NOT_FOUND'
            }), 404
        
        return jsonify({
            'reliability': class_data['class_data'][class_index]['reliability'],
            'precision_recall': class_data['class_data'][class_index]['precision_recall'],
            'roc_auc': class_data['class_data'][class_index]['roc_auc'],
            'roc_delta': class_data['class_data'][class_index].get('roc_delta'),
            'class_index': class_index,
            'model_key': model_key,
            'total_classes': class_data['n_classes'],
            'available_models': available_models,
            'success': True
        })
        
    except Exception as e:
        print(f"Error loading class-specific results: {e}")
        return jsonify({
            'error': 'Error loading class-specific results.',
            'code': 'INTERNAL_ERROR'
        }), 500

def export(jobid, class_index=None):
    """Export the results CSV - subset by class if specified"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/report.csv'):
        abort(400)
        return

    # Get class_index from query parameters if not passed as argument
    if class_index is None:
        class_index = request.args.get('class_index')

    try:
        df = pd.read_csv(folder + '/report.csv')
    except Exception as e:
        abort(500)
        return
    
    # Filter by class_index if specified
    if class_index is not None:
        try:
            class_index = int(class_index)
            original_count = len(df)
            df = df[df['class_index'] == class_index]
            filename = f'class_{class_index}_report.csv'
        except (ValueError, KeyError) as e:
            filename = 'report.csv'
    else:
        filename = 'report.csv'
    
    # Apply the same column dropping as before, but handle missing columns gracefully
    columns_to_drop = ['test_fpr', 'test_tpr', 'generalization_fpr', 'generalization_tpr', 'fop', 'mpv', 'precision', 'recall']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        df_filtered = df.drop(existing_columns_to_drop, axis=1)
    else:
        df_filtered = df
    
    return Response(
        df_filtered.to_csv(index=False),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename={filename}'}
    )

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

    if model_key is None:
        abort(400, description="model_key parameter is required")
        return

    if class_index is not None:
        # Extract from OvR models archive
        archive_path = folder + '/models/ovr_models.tar.gz'
        model_filename = f"{model_key}_ovr_class_{class_index}.joblib"
        zip_filename = f'ovr_class_{class_index}_model.zip'
    else:
        # Extract from main models archive
        archive_path = folder + '/models/main_models.tar.gz'
        model_filename = f"{model_key}.joblib"
        zip_filename = 'model.zip'

    if not os.path.exists(archive_path):
        abort(400, description=f"Model archive not found: {archive_path}")
        return

    # Extract specific model to temp location
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Extract the specific model file
                try:
                    if class_index is not None:
                        member_path = f"ovr_models/{model_filename}"
                    else:
                        member_path = f"main_models/{model_filename}"
                    
                    member = tar.getmember(member_path)
                    tar.extract(member, temp_dir)
                    extracted_path = f"{temp_dir}/{member.name}"
                except KeyError:
                    abort(404, description=f"Model not found in archive: {model_filename}")
                    return
            
            # Update threshold in predict.py
            with open('client/predict.py', 'r+') as file:
                contents = file.read()
                contents = re.sub(r'THRESHOLD = [\d.]+', 'THRESHOLD = ' + str(threshold), contents)
                file.seek(0)
                file.truncate()
                file.write(contents)

            # Copy extracted model to client directory
            copyfile(extracted_path, 'client/pipeline.joblib')
            copyfile(folder + '/input.csv', 'client/input.csv')

            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                files = os.listdir('client')
                for individualFile in files:
                    filePath = os.path.join('client', individualFile)
                    zf.write(filePath, individualFile)
            memory_file.seek(0)

            return send_file(memory_file, attachment_filename=zip_filename, as_attachment=True, cache_timeout=-1)
    except Exception as e:
        abort(500, description=f"Error extracting model: {str(e)}")
        return

def star_models(jobid):
    """Marks the selected models as starred"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    
    if os.path.exists(folder + '/starred.json'):
        with open(folder + '/starred.json') as starred_file:
            starred = json.load(starred_file)
    else:
        starred = []

    starred = list(set(starred + request.get_json()['models']))

    with open(folder + '/starred.json', 'w') as starred_file:
        json.dump(starred, starred_file)

    return jsonify({'success': True})

def un_star_models(jobid):
    """Removes the selected models as starred models"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    if not os.path.exists(folder + '/starred.json'):
        return jsonify({'success': True})

    with open(folder + '/starred.json') as starred_file:
        starred = json.load(starred_file)

    for item in request.get_json()['models']:
        try:
            starred.remove(item)
        except Exception:
            continue

    with open(folder + '/starred.json', 'w') as starred_file:
        json.dump(starred, starred_file)

    return jsonify({'success': True})

def get_starred(jobid):
    """Get the starred models for a job"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    if not os.path.exists(folder + '/starred.json'):
        return abort(404)

    with open(folder + '/starred.json') as starred_file:
        starred = json.load(starred_file)

    return jsonify(starred)
    
def list_available_models(jobid, class_index=None):
    """List models available in archives"""

    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]

    try:
        if class_index is not None:
            archive_path = folder + '/models/ovr_models.tar.gz'
            prefix = "ovr_models/"
            suffix = f"_ovr_class_{class_index}.joblib"
        else:
            archive_path = folder + '/models/main_models.tar.gz'
            prefix = "main_models/"
            suffix = ".joblib"
        
        if not os.path.exists(archive_path):
            return jsonify({
                'models': [],
                'count': 0,
                'archive_exists': False
            })
        
        models = []
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith(prefix) and member.name.endswith(suffix):
                    model_key = member.name[len(prefix):-len(suffix)]
                    models.append(model_key)
        
        return jsonify({
            'models': models,
            'count': len(models),
            'archive_exists': True,
            'class_index': class_index
        })
        
    except Exception as e:
        print(f"Error listing available models: {e}")
        return jsonify({
            'error': 'Error retrieving available models from archives.',
            'code': 'INTERNAL_ERROR'
        }), 500
    
def get_available_class_models(jobid):
    """Get list of models that have class-specific results"""
    
    if g.uid is None:
        abort(401)
        return

    folder = 'data/users/' + g.uid + '/jobs/' + jobid.urn[9:]
    class_results_dir = folder + '/class_results'

    try:
        available_models = get_available_models_with_class_results(class_results_dir)
        
        return jsonify({
            'models': available_models,
            'count': len(available_models),
            'has_class_results': len(available_models) > 0
        })
        
    except Exception as e:
        print(f"Error getting available class models: {e}")
        return jsonify({
            'error': 'Error retrieving available models with class-specific results.',
            'code': 'INTERNAL_ERROR'
        }), 500
