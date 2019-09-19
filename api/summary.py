"""
Generate final summary
"""

import pandas as pd

from .utils import model_key_to_name

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

def print_summary(results):
    """Prints the final summary"""

    if not len(results):
        return

    results = sorted(results.items(),
                     key=lambda x: (x[1]['auc'], x[1]['sensitivity'], x[1]['f1']), reverse=True)
    columns = list(results[0][1].keys())

    data = []
    runs = []

    for key, value in results:
        runs.append(model_key_to_name(key))
        data.append(list(value.values()))

    print('Best model:', runs[0], '\n')

    print('General summary')
    summary = pd.DataFrame(data, index=runs, columns=columns)

    print(summary)
