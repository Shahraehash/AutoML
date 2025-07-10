"""
Binary Classification Classifier

This module provides the BinaryClassifier class for handling binary classification tasks.
It extends the AutoMLClassifier base class with binary-specific logic.
"""

import numpy as np
import pandas as pd
import json
import os
from timeit import default_timer as timer
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve
from nyoka import skl_to_pmml, xgboost_to_pmml


from .base_classifier import AutoMLClassifier
from .utils.preprocess import preprocess
from .utils.utils import model_key_to_name, decimate_points, explode_key
from .utils.stats import clopper_pearson, roc_auc_ci, ppv_95_ci, npv_95_ci
from .utils.import_data import import_data
from .processors.estimators import ESTIMATORS, get_xgb_classifier
from .processors.scalers import SCALERS
from .processors.feature_selection import FEATURE_SELECTORS



class BinaryClassifier(AutoMLClassifier):
    """
    Handles binary classification tasks.
    
    This classifier is optimized for datasets with exactly 2 classes and provides
    binary-specific evaluation metrics and model handling.
    """
    
    def __init__(self, parameters, output_path='.', update_function=lambda x, y: None):
        """
        Initialize the Binary Classifier.
        
        Args:
            parameters (dict): Configuration parameters for the classifier
            output_path (str): Path where results and models will be saved
            update_function (callable): Callback function for progress updates
        """
        super().__init__(parameters, output_path, update_function)
    
    def import_csv(path, label_column, show_warning=False):
        """Import the specificed sheet"""

        # Read the CSV to memory and drop rows with empty values
        data = pd.read_csv(path)

        # Convert cell values to numerical data and drop invalid data
        data = data.apply(pd.to_numeric, errors='coerce').dropna()

        # Drop the label column from the data
        x = data.drop(label_column, axis=1)

        # Save the label colum values
        y = data[label_column]

        # Grab the feature names
        feature_names = list(x)

        # Convert to NumPy array
        x = x.to_numpy()

        # Get unique labels and label counts
        unique_labels = sorted(y.unique())

        label_counts = {}
        for label in unique_labels:
            label_counts[f'class_{int(label)}_count'] = data[data[label_column] == label].shape[0]
        
        # For backward compatibility with binary classification
        if show_warning:
            negative_count = label_counts.get('class_0_count', 0)
            positive_count = label_counts.get('class_1_count', 0)
            print('Negative Cases: %.7g\nPositive Cases: %.7g\n' % (negative_count, positive_count))
            if negative_count / positive_count < .9:
                print('Warning: Classes are not balanced.')
                
        return [x, y, feature_names, label_counts, 2]
    

    def generalize(self, pipeline, features, model, x2, y2, labels=None, threshold=.5, class_index=None):
        """"Generalize method"""
        # Process test data based on pipeline
        x2 = preprocess(features, pipeline, x2)
        proba = model.predict_proba(x2)
    
        probabilities = proba[:, 1]
        if threshold == .5:
            predictions = model.predict(x2)
        else:
            predictions = (probabilities >= threshold).astype(int)
            
        return self.generalization_report(labels, y2, predictions, probabilities, class_index)

    def generalization_report(self, labels, y2, predictions, probabilities, class_index=None):
        
        labels = ['Class 0', 'Class 1']
        print('\t', classification_report(y2, predictions, target_names=labels).replace('\n', '\n\t'))

        print('\tGeneralization:')
        accuracy = accuracy_score(y2, predictions)
        print('\t\tAccuracy:', accuracy)

        auc = roc_auc_score(y2, predictions)
        
        roc_auc = roc_auc_score(y2, probabilities)
        print('\t\tROC AUC:', roc_auc)
        
        _, tpr, _ = roc_curve(y2, probabilities)
        
        tn, fp, fn, tp = confusion_matrix(y2, predictions).ravel()
            
        mcc = matthews_corrcoef(y2, predictions)
        f1 = f1_score(y2, predictions)

        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        prevalence = (tp + fn) / (len(y2))

        return {
            'accuracy': round(accuracy, 4),
            'acc_95_ci': clopper_pearson(tp+tn, len(y2)),
            'mcc': round(mcc, 4),
            'avg_sn_sp': round(auc, 4),
            'roc_auc': round(roc_auc, 4),
            'roc_auc_95_ci': roc_auc_ci(roc_auc, tpr),
            'f1': round(f1, 4),
            'sensitivity': round(sensitivity, 4),
            'sn_95_ci': clopper_pearson(tp, tp+fn),
            'specificity': round(specificity, 4),
            'sp_95_ci': clopper_pearson(tn, tn+fp),
            'prevalence': round(prevalence, 4),
            'pr_95_ci': clopper_pearson(tp+fn, len(y2)),
            'ppv': round(tp / (tp+fp), 4) if tp+fp > 0 else 0,
            'ppv_95_ci': ppv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
            'npv': round(tn / (tn+fn), 4) if tn+fn > 0 else 0,
            'npv_95_ci': npv_95_ci(sensitivity, specificity, tp+fn, fp+tn, prevalence),
            'tn': int(tn),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp)
        }

    def generalize_model(self, payload, label, folder, threshold=.5):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')
        probabilities = pipeline.predict_proba(x)[:, 1]
        if threshold == .5:
            predictions = pipeline.predict(x)
        else:
            predictions = (probabilities >= threshold).astype(int)

        return self.generalization_report(['No ' + label, label], y, predictions, probabilities)

    def generate_pipeline(self, scaler, feature_selector, estimator, y_train, scoring=None, searcher='grid', shuffle=True, custom_hyper_parameters=None):
        """Generate the pipeline based on incoming arguments"""

        steps = []

        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        if feature_selector and FEATURE_SELECTORS[feature_selector]:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        steps.append(('debug', Debug()))

        if not scoring:
            scoring = ['accuracy']

        scorers = {}
        for scorer in scoring:
            scorers[scorer] = scorer

        search_step = SEARCHERS[searcher](estimator, scorers, shuffle, custom_hyper_parameters, y_train)

        steps.append(('estimator', search_step[0]))

        return (Pipeline(steps), search_step[1])

    def precision_recall(self, pipeline, features, model, x_test, y_test):
        """Compute precision recall curve"""

        # Transform values based on the pipeline
        x_test = preprocess(features, pipeline, x_test)
        
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(x_test)
            # For binary classification, decision_function returns 1D array
            if scores.ndim == 1:
                # Use scores directly - they're already decision function values
                precision, recall, _ = precision_recall_curve(y_test, scores)
            else:
                # Some classifiers might return 2D even for binary
                precision, recall, _ = precision_recall_curve(y_test, scores[:, 1])

        else:
            # Use predict_proba (Random Forest, etc.)
            probabilities = model.predict_proba(x_test)
            precision, recall, _ = precision_recall_curve(y_test, probabilities[:, 1])
                
        # Apply decimation
        recall, precision = decimate_points(
            [round(num, 4) for num in list(recall)],
            [round(num, 4) for num in list(precision)]
        )

        return {
            'precision': list(precision),
            'recall': list(recall)
        }

    def reliability(self, pipeline, features, model, x_test, y_test, class_index=None):
        """Compute reliability curve and Briar score"""

        # Transform values based on the pipeline
        x_test = preprocess(features, pipeline, x_test)

        if hasattr(model, 'decision_function'):
            probabilities = model.decision_function(x_test)
            if np.count_nonzero(probabilities):
                if probabilities.max() - probabilities.min() == 0:
                    probabilities = [0] * len(probabilities)
                else:
                    probabilities = (probabilities - probabilities.min()) / \
                        (probabilities.max() - probabilities.min())
            fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')
            brier_score = brier_score_loss(y_test, probabilities)
            
        else:
            probabilities = model.predict_proba(x_test)[:, 1]
            fop, mpv = calibration_curve(y_test, probabilities, n_bins=10, strategy='uniform')
            brier_score = brier_score_loss(y_test, probabilities)
            
        return {
            'brier_score': round(brier_score, 4),
            'fop': [round(num, 4) for num in list(fop)],
            'mpv': [round(num, 4) for num in list(mpv)]
        }

    def roc(self, pipeline, features, model, x_test, y_test, class_index=None):
        """Generate the ROC values"""

        # Transform values based on the pipeline
        x_test = preprocess(features, pipeline, x_test)

        probabilities = model.predict_proba(x_test)
        predictions = model.predict(x_test)
        
        
        fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
        roc_auc = roc_auc_score(y_test, probabilities[:, 1])

        fpr, tpr = decimate_points(
            [round(num, 4) for num in list(fpr)],
            [round(num, 4) for num in list(tpr)]
        )

        return {
            'fpr': list(fpr),
            'tpr': list(tpr),
            'roc_auc': roc_auc
        }

    def find_best_model(self, train_set=None, test_set=None, labels=None, label_column=None, parameters=None, output_path='.', update_function=lambda x, y: None):

        start = timer()     
        
        ignore_estimator = [x.strip() for x in parameters.get('ignore_estimator', '').split(',')]
        ignore_feature_selector = \
            [x.strip() for x in parameters.get('ignore_feature_selector', '').split(',')]
        ignore_scaler = [x.strip() for x in parameters.get('ignore_scaler', '').split(',')]
        ignore_searcher = [x.strip() for x in parameters.get('ignore_searcher', '').split(',')]
        shuffle = False if parameters.get('ignore_shuffle', '') != '' else True
        scorers = [x for x in SCORER_NAMES if x not in \
            [x.strip() for x in parameters.get('ignore_scorer', '').split(',')]]
        """Generates all possible models and outputs the generalization results using new class-based classifiers"""

        # Basic validation
        if train_set is None:
            print('Missing training data')
            return {}

        if test_set is None:
            print('Missing test data')
            return {}

        if label_column is None:
            print('Missing column name for classifier target')
            return {}

        custom_hyper_parameters = json.loads(parameters['hyper_parameters'])\
            if 'hyper_parameters' in parameters else None

        # Import data
        (x_train, x_val, y_train, y_val, x_test, y_test, feature_names, metadata) = \
            import_data(train_set, test_set, label_column)

        # Determine number of classes to choose appropriate classifier
        n_classes = len(np.unique(y_train))
        print(f'Detected {n_classes} classes in target column')

        total_fits = {}
        csv_header_written = False

        # Memory-efficient model storage
        main_models = {}  # Store main models in memory

        all_pipelines = list(itertools.product(*[
            filter(lambda x: False if x in ignore_estimator else True, ESTIMATOR_NAMES),
            filter(lambda x: False if x in ignore_scaler else True, SCALER_NAMES),
            filter(lambda x: False if x in ignore_feature_selector else True, FEATURE_SELECTOR_NAMES),
            filter(lambda x: False if x in ignore_searcher else True, SEARCHER_NAMES),
        ]))

        if not len(all_pipelines):
            print('No pipelines to run with the current configuration')
            return False

        report = open(output_path + '/report.csv', 'w+')
        report_writer = csv.writer(report)

        performance_report = open(output_path + '/performance_report.csv', 'w+')
        performance_report_writer = csv.writer(performance_report)
        performance_report_writer.writerow(['key', 'train_time (s)'])

        for index, (estimator, scaler, feature_selector, searcher) in enumerate(all_pipelines):

            # Trigger a callback for task monitoring purposes
            update_function(index, len(all_pipelines))

            key = '__'.join([scaler, feature_selector, estimator, searcher])
            print('Generating ' + model_key_to_name(key))

            # Generate the pipeline
            pipeline = self.generate_pipeline(scaler, feature_selector, estimator, y_train, scorers, searcher, shuffle, custom_hyper_parameters)

            if not estimator in total_fits:
                total_fits[estimator] = 0
            total_fits[estimator] += pipeline[1]

            # Fit the pipeline
            model = self.generate_model(pipeline[0], feature_names, x_train, y_train)
            performance_report_writer.writerow([key, model['train_time']])

            for scorer in scorers:
                key += '__' + scorer
                candidates = self.refit_model(pipeline[0], model['features'], estimator, scorer, x_train, y_train)
                total_fits[estimator] += len(candidates)

                for position, candidate in enumerate(candidates):
                    result = {
                        'key': key + '__' + str(position),
                        'class_type': 'binary',
                        'class_index': None,
                        'scaler': SCALER_NAMES[scaler],
                        'feature_selector': FEATURE_SELECTOR_NAMES[feature_selector],
                        'algorithm': ESTIMATOR_NAMES[estimator],
                        'searcher': SEARCHER_NAMES[searcher],
                        'scorer': SCORER_NAMES[scorer],
                    }

                    print('\t#%d' % (position+1))
                    # Store main model in memory instead of saving immediately
                    main_models[result['key']] = candidate['best_estimator']

                    result.update(self.generalize(pipeline[0], model['features'], candidate['best_estimator'], x2, y2, labels))
                    result.update({
                        'selected_features': list(model['selected_features']),
                        'feature_scores': model['feature_scores'],
                        'best_params': candidate['best_params']
                    })
                    roc_auc = self.roc(pipeline[0], model['features'], candidate['best_estimator'], x_test, y_test)
                    result.update({
                        'test_fpr': roc_auc['fpr'],
                        'test_tpr': roc_auc['tpr'],
                        'training_roc_auc': roc_auc['roc_auc']
                    })
                    result['roc_delta'] = round(abs(result['roc_auc'] - result['training_roc_auc']), 4)
                    roc_auc = self.roc(pipeline[0], model['features'], candidate['best_estimator'], x2, y2)
                    result.update({
                        'generalization_fpr': roc_auc['fpr'],
                        'generalization_tpr': roc_auc['tpr']
                    })
                    result.update(self.reliability(pipeline[0], model['features'], candidate['best_estimator'], x2, y2))
                    result.update(self.precision_recall(pipeline[0], model['features'], candidate['best_estimator'], x2, y2))

                    # Write main model results first
                    if not csv_header_written:
                        report_writer.writerow(result.keys())
                        csv_header_written = True

                    report_writer.writerow(list([str(i) for i in result.values()]))

        train_time = timer() - start
        print('\tTotal run time is {:.4f} seconds'.format(train_time), '\n')
        performance_report_writer.writerow(['total', train_time])
    
        report.close()
        performance_report.close()
        print('Total fits generated', sum(total_fits.values()))
        print_summary(output_path + '/report.csv')
        
        self.save_model_archives(main_models, None, output_path)

        # Update the metadata and write it out
        metadata.update({
            'date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'fits': total_fits,
            'ovr_reoptimized': None,
            'ovr_models_count': None,
            'main_models_count': len(main_models),
            'n_classes': n_classes
        })
            

        if output_path != '.':
            metadata_path = output_path + '/metadata.json'
            # Check if metadata file exists and load existing data
            existing_metadata = {}
            if os.path.exists(metadata_path):
    
                with open(metadata_path, 'r') as metafile:
                    existing_metadata = json.load(metafile)
        
                # Update with new metadata
                existing_metadata.update(metadata)
            
                # Write the updated metadata
                with open(metadata_path, 'w') as metafile:
                    json.dump(existing_metadata, metafile, indent=2)
            
        return True


    @staticmethod
    def create_model(self, key, hyper_parameters, selected_features, dataset_path=None, label_column=None, output_path='.', threshold=.5):
        """Refits the requested model and pickles it for export"""

        if dataset_path is None:
            print('Missing dataset path')
            return {}

        if label_column is None:
            print('Missing column name for classifier target')
            return {}

        # Import data
        (x_train, _, y_train, _, x2, y2, features, _) = \
            import_data(dataset_path + '/train.csv', dataset_path + '/test.csv', label_column)

        # Get pipeline details from the key
        scaler, feature_selector, estimator, _, _ = explode_key(key)
        steps = []

        # Drop the unused features
        if 'pca-' not in feature_selector:
            for index, feature in reversed(list(enumerate(features))):
                if feature not in selected_features:
                    x_train = np.delete(x_train, index, axis=1)
                    x2 = np.delete(x2, index, axis=1)

        # Add the scaler, if used
        if scaler and SCALERS[scaler]:
            steps.append(('scaler', SCALERS[scaler]))

        # Add the feature transformer
        if 'pca-' in feature_selector:
            steps.append(('feature_selector', FEATURE_SELECTORS[feature_selector]))

        # Add the estimator with proper XGBoost configuration
        if estimator == 'gb':
            n_classes = len(pd.Series(y_train).unique())
            base_estimator = get_xgb_classifier(n_classes)
        else:
            base_estimator = ESTIMATORS[estimator]
        
        steps.append(('estimator', base_estimator.set_params(**hyper_parameters)))

        # Fit the pipeline using the same training data
        pipeline = Pipeline(steps)
        model = self.generate_model(pipeline, selected_features, x_train, y_train)

        # If the model is DNN or RF, attempt to swap the estimator for a pickled one
        if os.path.exists(output_path + '/models/' + key + '.joblib'):
            pickled_estimator = load(output_path + '/models/' + key + '.joblib')
            pipeline = Pipeline(pipeline.steps[:-1] + [('estimator', pickled_estimator)])

        # Binary classification labels
        labels = ['No ' + label_column, label_column]

        # Assess the model performance and store the results
        generalization_result = self.generalize(pipeline, model['features'], pipeline['estimator'], x2, y2, labels)
        with open(output_path + '/pipeline.json', 'w') as statsfile:
            json.dump(generalization_result, statsfile)

        # Dump the pipeline to a file
        dump(pipeline, output_path + '/pipeline.joblib')
        pd.DataFrame([selected_features]).to_csv(output_path + '/input.csv', index=False, header=False)

        # Export the model as a PMML
        try:
            if estimator == 'gb':
                xgboost_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
            else:
                skl_to_pmml(pipeline, selected_features, label_column, output_path + '/pipeline.pmml')
        except Exception:
            try:
                os.remove(output_path + '/pipeline.pmml')
            except OSError:
                pass

        return generalization_result

    @staticmethod
    def generalize_ensemble(self, total_models, job_folder, dataset_folder, label):
        x2, y2, feature_names, _, _ = self.import_csv(dataset_folder + '/test.csv', label)

        data = pd.DataFrame(x2, columns=feature_names)

        soft_result = self.predict_ensemble(total_models, data, job_folder, 'soft')
        hard_result = self.predict_ensemble(total_models, data, job_folder, 'hard')

        labels = ['No ' + label, label]
    
        return {
            'soft_generalization': self.generalization_report(labels, y2, soft_result['predicted'], soft_result['probability']),
            'hard_generalization': self.generalization_report(labels, y2, hard_result['predicted'], hard_result['probability'])
        }

    @staticmethod
    def additional_precision(self, payload, label, folder, class_index=None):
        """Return additional precision recall curve"""

        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.precision_recall(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)
    
    @staticmethod
    def additional_reliability(self, payload, label, folder, class_index=None):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.reliability(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)

    @staticmethod
    def additional_roc(self, payload, label, folder, class_index=None):
        data = pd.DataFrame(payload['data'], columns=payload['columns']).apply(pd.to_numeric, errors='coerce').dropna()
        x = data[payload['features']].to_numpy()
        y = data[label]

        pipeline = load(folder + '.joblib')

        return self.roc(pipeline, payload['features'], pipeline.steps[-1][1], x, y, class_index)


    @staticmethod
    def predict(self, data, path='.', threshold=.5):
        """Predicts against the provided data"""

        # Load the pipeline
        pipeline = load(path + '.joblib')

        data = pd.DataFrame(data).dropna().values

        probability = pipeline.predict_proba(data)
        if threshold == .5:
            predicted = pipeline.predict(data)
        else:
            predicted = (probability[:, 1] >= threshold).astype(int)

        return {
            'predicted': predicted.tolist(),
            'probability': [sublist[predicted[index]] for index, sublist in enumerate(probability.tolist())]
        }

    @staticmethod
    def predict_ensemble(self, total_models, data, path='.', vote_type='soft'):
        """Predicts against the provided data by creating an ensemble of the selected models"""

        probabilities = []
        predictions = []

        for x in range(total_models):
            pipeline = load(path + '/ensemble' + str(x) + '.joblib')

            with open(path + '/ensemble' + str(x) + '_features.json') as feature_file:
                features = json.load(feature_file)

            selected_data = data[features].dropna().to_numpy()
            probabilities.append(pipeline.predict_proba(selected_data))
            predictions.append(pipeline.predict(selected_data))

        predictions = np.asarray(predictions).T
        probabilities = np.average(np.asarray(probabilities), axis=0)

        if vote_type == 'soft':
            predicted = np.argmax(probabilities, axis=1)
        else:
            predicted = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions.astype('int')
            )

        return {
            'predicted': predicted.tolist(),
            'probability': [sublist[predicted[index]] for index, sublist in enumerate(probabilities.tolist())]
        }
    