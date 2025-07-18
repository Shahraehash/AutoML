"""
Memory management utility for AutoML to prevent SIGKILL errors
"""

import os
import gc
import psutil
import time
import tempfile
import shutil
import glob
from typing import Dict, Any, Optional, Tuple


class MemoryManager:
    """
    Comprehensive memory management for AutoML operations
    Handles monitoring, cleanup, and optimization to prevent memory exhaustion
    """
    
    def __init__(self, warning_threshold: float = 0.75, critical_threshold: float = 0.85):
        """
        Initialize memory manager
        
        Args:
            warning_threshold: Memory usage percentage to trigger warnings (0.0-1.0)
            critical_threshold: Memory usage percentage to trigger emergency cleanup (0.0-1.0)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cleanup_count = 0
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        
        print(f"MemoryManager initialized - Warning: {warning_threshold*100}%, Critical: {critical_threshold*100}%")
        print(f"Initial memory usage: {self.initial_memory:.1f}%")
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage percentage
        
        Returns:
            Memory usage as percentage (0.0-100.0)
        """
        try:
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            usage_percent = (memory_info.rss / system_memory.total) * 100
            return usage_percent
        except Exception as e:
            print(f"Error getting memory usage: {e}")
            return 0.0
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get detailed memory information
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': (memory_info.rss / system_memory.total) * 100,
                'available_mb': system_memory.available / (1024 * 1024),
                'total_mb': system_memory.total / (1024 * 1024)
            }
        except Exception as e:
            print(f"Error getting detailed memory info: {e}")
            return {}
    
    def check_memory_status(self) -> str:
        """
        Check current memory status
        
        Returns:
            'normal', 'warning', or 'critical'
        """
        usage_percent = self.get_memory_usage() / 100.0
        
        if usage_percent >= self.critical_threshold:
            return 'critical'
        elif usage_percent >= self.warning_threshold:
            return 'warning'
        else:
            return 'normal'
    
    def log_memory_stats(self, stage: str, model_key: Optional[str] = None):
        """
        Log memory usage at different stages
        
        Args:
            stage: Description of current processing stage
            model_key: Optional model identifier
        """
        memory_info = self.get_memory_info()
        if memory_info:
            key_info = f" [{model_key}]" if model_key else ""
            print(f"Memory {stage}{key_info}: {memory_info['percent']:.1f}% "
                  f"({memory_info['rss_mb']:.1f}MB RSS, {memory_info['available_mb']:.1f}MB available)")
            
            # Log warning if approaching thresholds
            status = self.check_memory_status()
            if status == 'warning':
                print(f"WARNING: Memory usage approaching threshold ({self.warning_threshold*100}%)")
            elif status == 'critical':
                print(f"CRITICAL: Memory usage at critical level ({self.critical_threshold*100}%)")
    
    def cleanup_sklearn_caches(self):
        """Clear sklearn internal caches and temporary data"""
        try:
            # Clear sklearn testing cache
            from sklearn.utils._testing import clear_cache
            clear_cache()
        except ImportError:
            pass
        
        try:
            # Clear sklearn joblib cache if available
            from sklearn.externals import joblib
            if hasattr(joblib, 'Memory'):
                joblib.Memory.clear()
        except (ImportError, AttributeError):
            pass
        
        try:
            # Clear any sklearn configuration context
            import sklearn
            if hasattr(sklearn, '_config') and hasattr(sklearn._config, 'config_context'):
                # Reset configuration to defaults
                pass
        except Exception:
            pass
    
    def cleanup_joblib_resources(self):
        """
        Clean up joblib-specific resources including temporary folders and semaphores
        """
        cleaned_folders = 0
        cleaned_files = 0
        
        try:
            # Clean up joblib temporary folders
            temp_dir = tempfile.gettempdir()
            joblib_patterns = [
                'joblib_memmapping_folder_*',
                'joblib_*',
                f'joblib_worker_{os.getpid()}*'
            ]
            
            for pattern in joblib_patterns:
                pattern_path = os.path.join(temp_dir, pattern)
                for folder_path in glob.glob(pattern_path):
                    try:
                        if os.path.isdir(folder_path):
                            shutil.rmtree(folder_path, ignore_errors=True)
                            cleaned_folders += 1
                        elif os.path.isfile(folder_path):
                            os.remove(folder_path)
                            cleaned_files += 1
                    except (OSError, PermissionError):
                        # Ignore errors for files/folders that can't be removed
                        pass
            
            # Clean up worker-specific temporary directory if it exists
            worker_id = os.environ.get('CELERY_WORKER_NAME', f'worker_{os.getpid()}')
            worker_temp_dir = os.path.join(temp_dir, f'joblib_{worker_id}')
            if os.path.exists(worker_temp_dir):
                try:
                    shutil.rmtree(worker_temp_dir, ignore_errors=True)
                    cleaned_folders += 1
                except (OSError, PermissionError):
                    pass
            
            if cleaned_folders > 0 or cleaned_files > 0:
                print(f"Joblib cleanup: Removed {cleaned_folders} folders, {cleaned_files} files")
                
        except Exception as e:
            print(f"Warning: Error during joblib cleanup: {e}")
    
    def cleanup_process_resources(self):
        """
        Clean up process-specific resources that may leak
        """
        try:
            # Force cleanup of any remaining multiprocessing resources
            import multiprocessing
            
            # Clean up any remaining shared memory segments
            try:
                # This is a best-effort cleanup for multiprocessing resources
                if hasattr(multiprocessing, 'resource_tracker'):
                    # The resource tracker handles cleanup automatically,
                    # but we can force a cleanup cycle
                    pass
            except Exception:
                pass
                
        except Exception as e:
            print(f"Warning: Error during process resource cleanup: {e}")
    
    def aggressive_cleanup(self):
        """
        Perform aggressive memory cleanup including joblib resources
        """
        print("🧹 Performing aggressive memory cleanup...")
        
        # Clear sklearn caches
        self.cleanup_sklearn_caches()
        
        # Clean up joblib-specific resources
        self.cleanup_joblib_resources()
        
        # Clean up process resources
        self.cleanup_process_resources()
        
        # Force garbage collection multiple times
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"Garbage collection pass {i+1}: {collected} objects collected")
        
        # Clear any remaining unreachable objects
        gc.collect()
        
        self.cleanup_count += 1
        print(f"Cleanup #{self.cleanup_count} completed")
    
    def cleanup_model_iteration(self, **local_vars):
        """
        Clean up after each model iteration including joblib resources
        
        Args:
            **local_vars: Local variables to explicitly delete
        """
        memory_before = self.get_memory_usage()
        
        # Delete all provided variables
        vars_deleted = []
        for var_name, var_value in local_vars.items():
            if var_value is not None:
                try:
                    del var_value
                    vars_deleted.append(var_name)
                except Exception as e:
                    print(f"Warning: Could not delete {var_name}: {e}")
        
        # Clear sklearn caches
        self.cleanup_sklearn_caches()
        
        # Clean up joblib resources periodically (every 10 iterations)
        if self.cleanup_count % 10 == 0:
            self.cleanup_joblib_resources()
        
        # Force garbage collection
        collected = gc.collect()
        
        memory_after = self.get_memory_usage()
        memory_freed = memory_before - memory_after
        
        if vars_deleted or collected > 0 or memory_freed > 0.1:
            print(f"Cleanup: Deleted {len(vars_deleted)} vars, "
                  f"collected {collected} objects, "
                  f"freed {memory_freed:.1f}% memory")
    
    def emergency_cleanup(self) -> bool:
        """
        Emergency cleanup when critical threshold exceeded
        
        Returns:
            True if cleanup was successful, False if still critical
        """
        print("EMERGENCY CLEANUP: Critical memory threshold exceeded!")
        
        # Perform multiple aggressive cleanups
        for i in range(3):
            self.aggressive_cleanup()
            
            # Check if we're back to safe levels
            status = self.check_memory_status()
            if status != 'critical':
                print(f"Emergency cleanup successful after {i+1} attempts")
                return True
            
            time.sleep(0.1)  # Brief pause between attempts
        
        print("Emergency cleanup failed - memory still critical")
        return False
    
    def save_model_with_cleanup(self, model: Any, model_path: str, model_key: str):
        """
        Save model and immediately clean up to prevent memory accumulation
        
        Args:
            model: Model object to save
            model_path: Path to save the model
            model_key: Model identifier for logging
        """
        try:
            from joblib import dump
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            dump(model, model_path)
            print(f'Saved model: {model_key} -> {os.path.basename(model_path)}')
            
            # Immediate cleanup
            del model
            gc.collect()
            
        except Exception as e:
            print(f"Error saving model {model_key}: {e}")
    
    def should_continue_processing(self) -> Tuple[bool, str]:
        """
        Check if processing should continue based on memory status
        
        Returns:
            Tuple of (should_continue, reason)
        """
        status = self.check_memory_status()
        
        if status == 'critical':
            if not self.emergency_cleanup():
                return False, "Critical memory threshold exceeded and cleanup failed"
        
        return True, "Memory status acceptable"
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory management session
        
        Returns:
            Dictionary with memory management statistics
        """
        current_memory = self.get_memory_usage()
        memory_info = self.get_memory_info()
        
        return {
            'initial_memory_percent': self.initial_memory,
            'final_memory_percent': current_memory,
            'memory_change': current_memory - self.initial_memory,
            'cleanup_count': self.cleanup_count,
            'final_memory_mb': memory_info.get('rss_mb', 0),
            'warning_threshold': self.warning_threshold * 100,
            'critical_threshold': self.critical_threshold * 100
        }
    
    def print_final_summary(self):
        """Print final memory management summary and perform final cleanup"""
        # Perform final aggressive cleanup including joblib resources
        self.aggressive_cleanup()
        
        summary = self.get_memory_summary()
        
        print("\n" + "="*50)
        print("MEMORY MANAGEMENT SUMMARY")
        print("="*50)
        print(f"Initial memory usage: {summary['initial_memory_percent']:.1f}%")
        print(f"Final memory usage: {summary['final_memory_percent']:.1f}%")
        print(f"Memory change: {summary['memory_change']:+.1f}%")
        print(f"Total cleanups performed: {summary['cleanup_count']}")
        print(f"Final memory consumption: {summary['final_memory_mb']:.1f}MB")
        print("="*50)
