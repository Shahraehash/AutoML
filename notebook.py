#%%
"""
Example application use inside of a notebook
"""

# Start of user configuration
import os

# Valid estimators: rf,mlp,gb,svm,knn,lr,nb
os.environ['IGNORE_ESTIMATOR'] = 'rf,mlp,gb,svm,knn,lr'

# Valid feature selectors: none,pca-70,pca-80,pca-90,
# rf-25,rf-50,rf-75,select-25,select-50,select-75
os.environ['IGNORE_FEATURE_SELECTOR'] = 'rf-25,select-25'

# Valid scalers: none,std,minmax
os.environ['IGNORE_SCALER'] = 'none'

# Valid searchers: grid,random
os.environ['IGNORE_SEARCHER'] = 'grid'

# Valid scorers: f1_macro,roc_auc,accuracy
os.environ['IGNORE_SCORER'] = 'f1_macro'

# If the below line is uncommented, shuffling will be turned off
#os.environ['IGNORE_SHUFFLE'] = 'true'

# Change this to point to the train data
TRAIN_SET = 'sample-data/train.csv'

# Change this to point to the generalizatin data
TEST_SET = 'sample-data/test.csv'

# Change this to the columns name which identifies
# the label column
LABEL_COLUMN = 'AKI'

# End of user configuration

#%%
from api import api

LABELS = ['No ' + LABEL_COLUMN, LABEL_COLUMN]
api.find_best_model(TRAIN_SET, TEST_SET, LABELS, LABEL_COLUMN)
