"""
AutoML Classifiers Module

This module provides a class-based architecture for different types of classification tasks:
- Binary Classification
- Multiclass Classification with Macro-Averaging  
- One-vs-Rest (OvR) Classification

Each classifier implements specific evaluation methods for their classification type,
eliminating the need for conditional logic in evaluation functions.
"""

from .base_classifier import AutoMLClassifier
from .binary_classifier import BinaryClassifier
from .multiclass_macro_classifier import MulticlassMacroClassifier
from .multiclass_ovr_classifier import OvRClassifier
from .ensemble_model_classifier import EnsembleModelClassifier
from .static_model_classifier import StaticModelClassifier
from .tandem_model_classifier import TandemModelClassifier

__all__ = [
    'AutoMLClassifier',
    'BinaryClassifier', 
    'MulticlassMacroClassifier',
    'OvRClassifier',
    'EnsembleModelClassifier',
    'StaticModelClassifier',
    'TandemModelClassifier',
]
