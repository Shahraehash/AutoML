"""
Generate final summary
"""

import os

import pandas as pd

from .utils import model_key_to_name

def print_summary(results):
    """Prints the final summary"""
    print(f"In print summary {results} {os.path.exists(results)}")
    if not os.path.exists(results):
        return

    print(f"Csv file has columns {pd.read_csv(results).columns}")
    print(f"key generated is {pd.read_csv(results).sort_values(by=['avg_sn_sp', 'sensitivity', 'f1'], ascending=False).iloc[0]['key']}")

    print('Best model:', model_key_to_name(
        pd.read_csv(results).sort_values(
            by=['avg_sn_sp', 'sensitivity', 'f1'], ascending=False
        ).iloc[0]['key']
    ), '\n')
