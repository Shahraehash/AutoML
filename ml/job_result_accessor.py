"""
Job Result Accessor

This module provides a clean abstraction layer for accessing saved job results,
including CSV files, model archives, and class-specific data. It centralizes
all file operations that were previously scattered across API functions.
"""

import json
import os
import tarfile
import tempfile
import zipfile
import gzip
import pickle
from io import BytesIO
import pandas as pd
from joblib import load
from shutil import copyfile, rmtree
import re


class ResultNotFoundError(Exception):
    """Exception raised when requested results are not found."""
    def __init__(self, message, code='RESULT_NOT_FOUND', status_code=404):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ModelNotFoundError(Exception):
    """Exception raised when requested model is not found."""
    def __init__(self, message, code='MODEL_NOT_FOUND', status_code=404):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class JobResultAccessor:
    """
    Provides clean access to job results stored in files.
    
    This class centralizes all file operations for accessing:
    - Main results from CSV files
    - Class-specific results from .pkl.gz files
    - Models from compressed archives
    - Metadata and configuration
    """
    
    def __init__(self, job_folder):
        """
        Initialize the accessor for a specific job folder.
        
        Args:
            job_folder (str): Path to the job folder containing results
        """
        self.job_folder = job_folder
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load job metadata from metadata.json file."""
        metadata_path = f'{self.job_folder}/metadata.json'
        if not os.path.exists(metadata_path):
            raise ResultNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ResultNotFoundError(f"Error loading metadata: {str(e)}")
    
    def get_main_results(self):
        """
        Get main results from the report.csv file.
        
        Returns:
            dict: Dictionary containing results and metadata
        """
        csv_path = f'{self.job_folder}/report.csv'
        if not os.path.exists(csv_path):
            raise ResultNotFoundError("No results available for this job", 'NO_RESULTS', 400)
        
        try:
            data = json.loads(pd.read_csv(csv_path).to_json(orient='records'))
            return {
                'results': data,
                'metadata': self.metadata
            }
        except (ValueError, pd.errors.EmptyDataError) as e:
            raise ResultNotFoundError(f"Error reading results: {str(e)}", 'INVALID_RESULTS', 400)
    
    def get_class_results(self, class_index, model_key=None):
        """
        Get class-specific results for multiclass models.
        
        Args:
            class_index (int): Index of the class
            model_key (str, optional): Specific model key, uses first available if None
            
        Returns:
            dict: Class-specific results including reliability, precision_recall, roc_auc
        """
        class_results_dir = f'{self.job_folder}/class_results'
        
        try:
            available_models = self._get_available_models_with_class_results(class_results_dir)
            
            if not available_models:
                raise ResultNotFoundError(
                    'No class-specific results available for this job.',
                    'NO_CLASS_RESULTS',
                    400
                )
            
            # Use first available model if no model_key specified
            if not model_key:
                model_key = available_models[0]
            
            class_data = self._load_class_results(class_results_dir, model_key)
            
            if not class_data:
                raise ModelNotFoundError(f'Class-specific results not found for model {model_key}.')
            
            class_index = int(class_index)
            
            if class_index not in class_data['class_data']:
                raise ResultNotFoundError(f'Class {class_index} not found in results.', 'CLASS_NOT_FOUND')
            
            return {
                'reliability': class_data['class_data'][class_index]['reliability'],
                'precision_recall': class_data['class_data'][class_index]['precision_recall'],
                'roc_auc': class_data['class_data'][class_index]['roc_auc'],
                'roc_delta': class_data['class_data'][class_index].get('roc_delta'),
                'class_index': class_index,
                'model_key': model_key,
                'total_classes': class_data['n_classes'],
                'available_models': available_models,
                'success': True
            }
            
        except ValueError:
            raise ResultNotFoundError('Invalid class index provided.', 'INVALID_CLASS_INDEX', 400)
        except Exception as e:
            if isinstance(e, (ResultNotFoundError, ModelNotFoundError)):
                raise
            raise ResultNotFoundError(f'Error loading class-specific results: {str(e)}', 'INTERNAL_ERROR', 500)
    
    def _load_class_results(self, class_results_dir, model_key):
        """Load class-specific results from .pkl.gz file."""
        filepath = f"{class_results_dir}/{model_key}.pkl.gz"
        if os.path.exists(filepath):
            try:
                with gzip.open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading class results for {model_key}: {e}")
                return None
        return None
    
    def _get_available_models_with_class_results(self, class_results_dir):
        """Get list of models that have class-specific results."""
        if not os.path.exists(class_results_dir):
            return []
        
        models = []
        try:
            for filename in os.listdir(class_results_dir):
                if filename.endswith('.pkl.gz'):
                    models.append(filename.replace('.pkl.gz', ''))
        except Exception as e:
            print(f"Error listing class results: {e}")
            return []
        
        return models
    
    def list_available_models(self, class_index=None):
        """
        List models available in archives.
        
        Args:
            class_index (int, optional): If provided, list OvR models for this class
            
        Returns:
            dict: Information about available models
        """
        try:
            if class_index is not None:
                archive_path = f'{self.job_folder}/models/ovr_models.tar.gz'
                prefix = "ovr_models/"
                suffix = f"_ovr_class_{class_index}.joblib"
            else:
                archive_path = f'{self.job_folder}/models/main_models.tar.gz'
                prefix = "main_models/"
                suffix = ".joblib"
            
            if not os.path.exists(archive_path):
                return {
                    'models': [],
                    'count': 0,
                    'archive_exists': False
                }
            
            models = []
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.startswith(prefix) and member.name.endswith(suffix):
                        model_key = member.name[len(prefix):-len(suffix)]
                        models.append(model_key)
            
            return {
                'models': models,
                'count': len(models),
                'archive_exists': True,
                'class_index': class_index
            }
            
        except Exception as e:
            print(f"Error listing available models: {e}")
            raise ResultNotFoundError(
                'Error retrieving available models from archives.',
                'INTERNAL_ERROR',
                500
            )
    
    def get_available_class_models(self):
        """Get list of models that have class-specific results."""
        class_results_dir = f'{self.job_folder}/class_results'
        
        try:
            available_models = self._get_available_models_with_class_results(class_results_dir)
            
            return {
                'models': available_models,
                'count': len(available_models),
                'has_class_results': len(available_models) > 0
            }
            
        except Exception as e:
            print(f"Error getting available class models: {e}")
            raise ResultNotFoundError(
                'Error retrieving available models with class-specific results.',
                'INTERNAL_ERROR',
                500
            )
    
    def get_model_from_archive(self, model_key, class_index=None):
        """
        Extract a specific model from compressed archives.
        
        Args:
            model_key (str): Key identifying the model
            class_index (int, optional): Class index for OvR models
            
        Returns:
            object: Loaded model object
        """
        if model_key is None:
            raise ModelNotFoundError("model_key parameter is required")

        if class_index is not None:
            # Extract from OvR models archive
            archive_path = f'{self.job_folder}/models/ovr_models.tar.gz'
            model_filename = f"{model_key}_ovr_class_{class_index}.joblib"
        else:
            # Extract from main models archive
            archive_path = f'{self.job_folder}/models/main_models.tar.gz'
            model_filename = f"{model_key}.joblib"

        if not os.path.exists(archive_path):
            raise ModelNotFoundError(f"Model archive not found: {archive_path}")

        # Extract specific model to temp location
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(archive_path, "r:gz") as tar:
                    try:
                        if class_index is not None:
                            member_path = f"ovr_models/{model_filename}"
                        else:
                            member_path = f"main_models/{model_filename}"
                        
                        member = tar.getmember(member_path)
                        tar.extract(member, temp_dir)
                        extracted_path = f"{temp_dir}/{member.name}"
                    except KeyError:
                        raise ModelNotFoundError(f"Model not found in archive: {model_filename}")
                
                # Load the model
                return load(extracted_path)
        
        except Exception as e:
            if isinstance(e, ModelNotFoundError):
                raise
            raise ModelNotFoundError(f"Error extracting model: {str(e)}")
    
    def create_model_export_zip(self, model_key, class_index=None, threshold=0.5):
        """
        Create a ZIP file for model export.
        
        Args:
            model_key (str): Key identifying the model
            class_index (int, optional): Class index for OvR models
            threshold (float): Prediction threshold
            
        Returns:
            BytesIO: ZIP file in memory
        """
        if class_index is not None:
            zip_filename = f'ovr_class_{class_index}_model.zip'
        else:
            zip_filename = 'model.zip'

        # Extract model to temp location
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get the model from archive
                if class_index is not None:
                    archive_path = f'{self.job_folder}/models/ovr_models.tar.gz'
                    model_filename = f"{model_key}_ovr_class_{class_index}.joblib"
                else:
                    archive_path = f'{self.job_folder}/models/main_models.tar.gz'
                    model_filename = f"{model_key}.joblib"

                if not os.path.exists(archive_path):
                    raise ModelNotFoundError(f"Model archive not found: {archive_path}")

                with tarfile.open(archive_path, "r:gz") as tar:
                    try:
                        if class_index is not None:
                            member_path = f"ovr_models/{model_filename}"
                        else:
                            member_path = f"main_models/{model_filename}"
                        
                        member = tar.getmember(member_path)
                        tar.extract(member, temp_dir)
                        extracted_path = f"{temp_dir}/{member.name}"
                    except KeyError:
                        raise ModelNotFoundError(f"Model not found in archive: {model_filename}")
                
                # Update threshold in predict.py
                with open('client/predict.py', 'r+') as file:
                    contents = file.read()
                    contents = re.sub(r'THRESHOLD = [\d.]+', 'THRESHOLD = ' + str(threshold), contents)
                    file.seek(0)
                    file.truncate()
                    file.write(contents)

                # Copy extracted model to client directory
                copyfile(extracted_path, 'client/pipeline.joblib')
                copyfile(f'{self.job_folder}/input.csv', 'client/input.csv')

                # Create ZIP file
                memory_file = BytesIO()
                with zipfile.ZipFile(memory_file, 'w') as zf:
                    files = os.listdir('client')
                    for individual_file in files:
                        file_path = os.path.join('client', individual_file)
                        zf.write(file_path, individual_file)
                memory_file.seek(0)

                return memory_file
        
        except Exception as e:
            if isinstance(e, ModelNotFoundError):
                raise
            raise ModelNotFoundError(f"Error creating model export: {str(e)}")
    
    def get_csv_export_data(self, class_index=None):
        """
        Get CSV data for export, optionally filtered by class.
        
        Args:
            class_index (int, optional): Filter results by class index
            
        Returns:
            tuple: (filtered_dataframe, filename)
        """
        csv_path = f'{self.job_folder}/report.csv'
        if not os.path.exists(csv_path):
            raise ResultNotFoundError("No results available for export", 'NO_RESULTS', 400)
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ResultNotFoundError(f"Error reading results for export: {str(e)}", 'INVALID_RESULTS', 500)
        
        # Filter by class_index if specified
        if class_index is not None:
            try:
                class_index = int(class_index)
                df = df[df['class_index'] == class_index]
                filename = f'class_{class_index}_report.csv'
            except (ValueError, KeyError):
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
        
        return df_filtered, filename
    
    def get_starred_models(self):
        """Get the list of starred models for this job."""
        starred_path = f'{self.job_folder}/starred.json'
        if not os.path.exists(starred_path):
            raise ResultNotFoundError("No starred models found", 'NO_STARRED_MODELS', 404)
        
        try:
            with open(starred_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ResultNotFoundError(f"Error loading starred models: {str(e)}", 'INVALID_STARRED_DATA', 500)
    
    def update_starred_models(self, models_to_add=None, models_to_remove=None):
        """
        Update the starred models list.
        
        Args:
            models_to_add (list, optional): Models to add to starred list
            models_to_remove (list, optional): Models to remove from starred list
        """
        starred_path = f'{self.job_folder}/starred.json'
        
        # Load existing starred models
        if os.path.exists(starred_path):
            try:
                with open(starred_path, 'r') as f:
                    starred = json.load(f)
            except (json.JSONDecodeError, IOError):
                starred = []
        else:
            starred = []
        
        # Add new models
        if models_to_add:
            starred = list(set(starred + models_to_add))
        
        # Remove models
        if models_to_remove:
            for item in models_to_remove:
                try:
                    starred.remove(item)
                except ValueError:
                    continue
        
        # Save updated list
        try:
            with open(starred_path, 'w') as f:
                json.dump(starred, f)
        except IOError as e:
            raise ResultNotFoundError(f"Error saving starred models: {str(e)}", 'SAVE_ERROR', 500)
    
    def file_exists(self, filename):
        """Check if a specific file exists in the job folder."""
        return os.path.exists(f'{self.job_folder}/{filename}')
    
    def get_file_path(self, filename):
        """Get the full path to a file in the job folder."""
        return f'{self.job_folder}/{filename}'
    
    def cleanup_results(self):
        """Clean up class results and other temporary files for this job."""
        class_results_dir = f'{self.job_folder}/class_results'
        if os.path.exists(class_results_dir):
            try:
                rmtree(class_results_dir)
                print(f"Cleaned up class results directory: {class_results_dir}")
            except Exception as e:
                print(f"Error cleaning up class results: {e}")
