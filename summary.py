import pandas as pd
import numpy as np

from utils import modelKeyToName

def printSummary(results):
    data = []
    columns = []
    runs = []
    scores = []

    for key in results:
        columns = list(results[key].keys())
        runs.append(modelKeyToName(key))
        values = list(results[key].values())
        scores.append(sum([1-x for x in values]))
        data.append(values)

    print('Best model:', runs[np.array(scores).argmin()], '\n')

    print('General summary (%d models generated):' % (len(results) * 10))
    summary = pd.DataFrame(data, index=runs, columns=columns)
    summary.to_csv('report.csv')

    print(summary)