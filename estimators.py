# Dependencies
from functools import partial

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

estimators = {
    'gb': GradientBoostingClassifier,
    'knn': KNeighborsClassifier,
    'lr': partial(LogisticRegression, solver='lbfgs', max_iter=1000),
    'mlp': MLPClassifier,
    'nb': GaussianNB,
    'rf': partial(RandomForestClassifier, n_estimators=10),
    'svm': partial(SVC, gamma='auto')
}

estimatorNames = {
    'gb': 'gradient boosting machine',
    'knn': 'K-nearest neighbor',
    'lr': 'logistic regression',
    'mlp': 'neural network',
    'nb': 'naive Bayes',
    'rf': 'random forest',
    'svm': 'support vector machine'
}
