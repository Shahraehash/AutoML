"""
Define hyper-parameter ranges for grid search
"""

HYPER_PARAMETER_RANGE = {
    'gb': {
        'learning_rate': [0.1, 0.05, 0.02, 0.01],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [20, 50, 100, 150],
        'max_features': [1.0, 0.3, 0.1]
    },
    'knn': [
        {
            'n_neighbors': list(range(1, 31)),
            'weights': ['uniform']
        },
        {
            'n_neighbors': list(range(1, 31)),
            'weights': ['distance']
        }
    ],
    'lr': {
        'C': [.01, .1, 1, 2, 3, 4, 5, 10, 100],
        'solver': ['lbfgs'],
        'max_iter': [100, 200, 500]
    },
    'mlp': [
        {
            'max_iter': [300, 400],
            'activation': ['tanh'],
            'learning_rate': ['constant', 'adaptive'],
            'alpha': [.0001, .0005],
            'tol': [.005, .0001],
            'hidden_layer_sizes': [(10,), (20,), (50,), (100,), (200,), (10, 10, 10)]
        },
        {
            'max_iter': [300, 400],
            'activation': ['relu'],
            'learning_rate': ['constant', 'adaptive'],
            'alpha': [.0001, .0005],
            'tol': [.005, .0001],
            'hidden_layer_sizes': [(10,), (20,), (50,), (100,), (200,), (10, 10, 10)]
        }
    ],
    'rf': {
        'bootstrap': [True],
        'max_depth': [50, 80, 110],
        'max_features': ['auto'],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'n_ESTIMATORS': [10, 100, 200, 300, 1000]
    },
    'svm': {
        'C': [.1, 1, 10, 100, 1000],
        'kernel': ['rbf'],
        'gamma': [1, .1, .5, .01, .05, .001, .005, .0001]
    }
}
