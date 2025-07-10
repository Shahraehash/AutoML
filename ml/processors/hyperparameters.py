"""
Define hyper-parameter ranges for grid search
"""

from scipy.stats import randint, norm, uniform

HYPER_PARAMETER_RANGE = {
    'grid': {
        'gb': {
            'learning_rate': [.01, .05, .1],
            'max_depth': [2, 3, 4, 5, 6],
            'n_estimators': [100, 1000]
        },
        'knn': [
            {
                'n_neighbors': list(range(3, 31)),
                'weights': ['uniform']
            },
            {
                'n_neighbors': list(range(3, 31)),
                'weights': ['distance']
            }
        ],
        'lr': {
            'C': [.01, .1, 1, 2, 3, 4, 5, 10, 100],
            'solver': ['lbfgs'],
            'max_iter': [100, 200, 500]
        },
        'mlp': {
            'activation': ['tanh', 'relu'],
            'learning_rate': ['constant', 'adaptive'],
            'alpha': [.01, .1, 1], 
            'tol': [.001, .005, .01],
            'hidden_layer_sizes': [(50,), (100,)]
        },
        'rf': {
            'bootstrap': [True],
            'max_depth': [50, 80, 110],
            'max_features': ['sqrt'],
            'min_samples_leaf': [3, 4, 5],
            'n_estimators': [10, 100, 200, 300, 1000]
        },
        'svm': {
            'C': [.1, 1, 10, 100, 1000],
            'kernel': ['rbf'],
            'gamma': [1, .1, .5, .01, .05, .001, .005, .0001]
        }
    },
    'random': {
        'gb': {
            'max_depth': range(2, 6),
            'n_estimators': range(100, 200),
            'learning_rate': uniform(0.01, 0.1)
        },
        'mlp': {
            'max_iter': [300, 400],
            'activation': ['tanh', 'relu'],
            'learning_rate': ['constant', 'adaptive'],
            'alpha': uniform(loc=0, scale=4),
            'tol': uniform(0.0001, 0.005),
            'hidden_layer_sizes': [(50,), (100,), (5, 5), (7, 7), (3, 3, 3), (5, 5, 5)],
            'n_iter_no_change': (3, 5, 10)
        },
        'rf': {
            'bootstrap': [True],
            'max_depth': randint(50, 110),
            'max_features': ['sqrt'],
            'min_samples_leaf': randint(3, 5),
            'min_samples_split': randint(2, 4),
            'n_estimators': randint(100, 1000)
        },
        'svm': {
            'C': norm(1, 0.1),
            'gamma': uniform(0.01, 0.1)
        },
        'knn': lambda class_member_count: {

            # `n_neighbors` should be an integer between 1 and
            # the smallest count of cases of a given class.
            'n_neighbors': list(range(3, class_member_count + 1)),
            'weights': ['distance', 'uniform']
        },
        'lr': {
            'C': uniform(loc=0, scale=4),
            'solver': ['lbfgs'],
            'max_iter': list(range(100, 500)),
            'tol': uniform(loc=0, scale=4)
        }
    }
}
