"""
Generate final summary
"""

import os

import pandas as pd

from .utils import model_key_to_name

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

def print_summary(results):
    """Prints the final summary"""

    if not os.path.exists(results):
        return

    print('Best model:', model_key_to_name(
        pd.read_csv(results).sort_values(
            by=['auc', 'sensitivity', 'f1'], ascending=False
        ).iloc[0]['key']
    ), '\n')
