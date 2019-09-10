import pandas as pd

from estimators import estimatorNames
from feature_selection import featureSelectorNames
from scalers import scalerNames
from scorers import scorerNames

def printSummary(results):
    data = columns = runs = []

    for key in results:
        columns = list(results[key].keys())
        runs.append(keyToName(key))
        data.append(list(results[key].values()))

    print('General summary (%d models generated):' % (len(results) * 10))
    print(pd.DataFrame(data, index=runs, columns=columns))

def keyToName(key):
    keys = key.split('__')
    return '%s (%s) w/ %s and %s' % (estimatorNames[keys[2]], scorerNames[keys[3]], scalerNames[keys[0]], featureSelectorNames[keys[1]])