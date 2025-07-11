"""
Defines all estimators used
"""

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

ESTIMATORS = {
    'gb': XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
    'knn': KNeighborsClassifier(),
    'lr': LogisticRegression(solver='lbfgs', max_iter=1000), #Learns multinomial not OvR
    'mlp': MLPClassifier(),
    'nb': GaussianNB(),
    'rf': RandomForestClassifier(n_estimators=10),
    'svm': SVC(gamma='auto', probability=True, decision_function_shape='ovr'),
}

def get_xgb_classifier(n_classes=2):
    """Get XGBClassifier with appropriate objective based on number of classes"""
    if n_classes > 2:
        return XGBClassifier(
            objective='multi:softprob', 
            eval_metric='mlogloss',
            num_class=n_classes  # Explicitly set num_class for multiclass
        )
    else:
        return XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss'
            # No num_class needed for binary classification
        )

ESTIMATOR_NAMES = {
    'gb': 'gradient boosting machine',
    'knn': 'K-nearest neighbor',
    'lr': 'logistic regression',
    'mlp': 'neural network',
    'nb': 'naive Bayes',
    'rf': 'random forest',
    'svm': 'support vector machine'
}
