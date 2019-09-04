# Define hyper-parameter ranges for grid search
hyperParameterRange = {
    'gb': {
        'gb__learning_rate': [0.1, 0.05, 0.02, 0.01],
        'gb__max_depth': [4, 6, 8],
        'gb__min_samples_leaf': [20, 50, 100, 150],
        'gb__max_features': [1.0, 0.3, 0.1] 
    },
    'knn': [
        {
            'knn__n_neighbors': list(range(1, 31)),
            'knn__weights': ['uniform']
        },
        {
            'knn__n_neighbors': list(range(1, 31)),
            'knn__weights': ['distance']
        }
    ],
    'lr': {
        'lr__C': [.01, .1, 1, 2, 3, 4, 5, 10, 100],
        'lr__solver': ['lbfgs'],
        'lr__max_iter': [100,200,500]
    },
    'mlp': [
        {
            'mlp__max_iter': [300, 400],
            'mlp__activation': ['tanh'], 
            'mlp__learning_rate': ['constant', 'adaptive'],
            'mlp__alpha': [.0001, .0005], 
            'mlp__tol': [.005, .0001],
            'mlp__hidden_layer_sizes': [(10,), (20,), (50,), (100,), (200,), (10,10,10)]
        },
        {
            'mlp__max_iter': [300, 400],
            'mlp__activation': ['relu'], 
            'mlp__learning_rate': ['constant', 'adaptive'],
            'mlp__alpha': [.0001, .0005], 
            'mlp__tol': [.005, .0001],
            'mlp__hidden_layer_sizes': [(10,), (20,), (50,), (100,), (200,), (10,10,10)]
        }
    ],
    'rf': {
        'rf__bootstrap': [True],
        'rf__max_depth': [50, 80, 110],
        'rf__max_features': ['auto'],
        'rf__min_samples_leaf': [3, 4, 5],
        'rf__min_samples_split': [2,3,4],
        'rf__n_estimators': [10, 100, 200, 300, 1000]
    },
    'svm': {
        'svm__C': [.1, 1, 10, 100, 1000],
        'svm__kernel': ['rbf'],
        'svm__gamma': [1, .1, .5, .01, .05, .001, .005, .0001]
    }
}