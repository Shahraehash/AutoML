# Dependencies
from functools import partial

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

estimators = {
    'gb': Pipeline([('gb', GradientBoostingClassifier())]),
    'knn': Pipeline([('knn', KNeighborsClassifier())]),
    'lr': Pipeline([('lr', LogisticRegression(solver='lbfgs', max_iter=1000))]),
    'mlp': Pipeline([('mlp', MLPClassifier())]),
    'nb': Pipeline([('nb', GaussianNB())]),
    'rf': Pipeline([('rf', RandomForestClassifier(n_estimators=10))]),
    'svm': Pipeline([('svm', SVC(gamma='auto'))]),
    'svm-scaled': Pipeline([('std_scaler', StandardScaler()), ('svm', SVC(gamma='auto'))])
}

estimatorNames = {
    'gb': 'gradient boosting machine',
    'knn': 'K-nearest neighbor',
    'lr': 'logistic regression',
    'mlp': 'neural network',
    'nb': 'naive Bayes',
    'rf': 'random forest',
    'svm': 'support vector machine',
    'svm-scaled': 'support vector machine with standard scaler'
}
