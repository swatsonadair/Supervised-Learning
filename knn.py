import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

from load_datasets import *
from classification import *

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier


class KNeighborsOptimizer(Classification):

    def __init__(self):
        self.method = 'knn'
        self.metric = 'euclidean'
        self.weights = 'uniform'
        self.n_neighbors = 5
        self.k_fold = 10

    def create_classifier(self, metric=None, weights=None, n_neighbors=None):

        metric = metric if metric is not None else self.metric
        weights = weights if weights is not None else self.weights
        n_neighbors = n_neighbors if n_neighbors is not None else self.n_neighbors
        
        clf = KNeighborsClassifier(
                metric=metric, 
                weights=weights, 
                n_neighbors=n_neighbors)

        return clf


    def compare_metric(self):

        print 'Compare metric'
        results = {
            'euclidean': {},
            'manhattan': {},
            'minkowski': {},
            'chebyshev': {}
        }

        for hyperparam in results:
            
            # Perform Training
            clf = self.create_classifier(metric=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(metric=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'metric', 'Metric')


    def compare_weights(self):

        print 'Compare Weights'
        results = {
            'uniform': {},
            'distance': {}
        }

        for hyperparam in results:

            # Perform Training
            clf = self.create_classifier(weights=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(weights=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'weights', 'Weights')


    def compare_n_neighbors(self):

        print 'Compare N-Neighbors'
        results = {}
        for i in range(1, 16, 2):
            results[i] = {}

        for hyperparam in results:

            # Perform Training
            clf = self.create_classifier(n_neighbors=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            start = time.time()
            y_pred = clf.predict(self.X)
            predict_end = time.time()
            predict_elapsed = predict_end - start
            results[hyperparam]['Predict Time'] = predict_elapsed

            # Perform K-fold cross validation
            clf = self.create_classifier(n_neighbors=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'n', 'N-Neighbors')
        self.plot_times(results, 'n', 'N-Neighbors')


def main():

    usage = "usage: %prog -c \n" 
    usage+= "use [c] to generate comparisons between hyperparameters"
    parser = OptionParser(usage=usage)
    parser.add_option("-c", "--compare", dest="compare", help="Generate comparisons between hyperparameters", action="store_true")
    (options, args) = parser.parse_args()

    #------- ABALONE ------#

    print 'Abalone'

    X, y = load_abalone()
    labels = np.array(np.unique(y))
    y = np.array([list(labels).index(v) for v in y])

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    opt = KNeighborsOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'a'
    opt.title = 'K-NN Classifier: Abalone'

    """params = {
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'n_neighbors': np.arange(1, 16, 2)
    }
    clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=10, scoring='f1_macro', return_train_score=True)
    clf.fit(X, y)

    best_score = clf.best_score_
    print best_score

    best_params = clf.best_params_
    print best_params
    
    opt.metric = best_params['metric']
    opt.weights = best_params['weights']
    opt.algorithm = best_params['algorithm']
    opt.n_neighbors = best_params['n_neighbors']"""

    opt.metric = 'manhattan'
    opt.weights = 'distance'
    opt.algorithm = 'ball_tree'
    opt.n_neighbors = 5

    if options.compare:
        opt.compare_metric()
        opt.compare_weights()
        opt.compare_n_neighbors()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()


    #------- LETTERS ------#

    print 'Letters'

    X, y = load_letters()
    labels = np.array(np.unique(y))
    y = np.array([list(labels).index(v) for v in y])

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    opt = KNeighborsOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'l'
    opt.title = 'K-NN Classifier: Letters'

    """params = {
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'n_neighbors': np.arange(1, 16, 2)
    }
    clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=10, scoring='f1_macro', return_train_score=True)
    clf.fit(X, y)

    best_score = clf.best_score_
    print best_score

    best_params = clf.best_params_
    print best_params

    opt.metric = best_params['metric']
    opt.weights = best_params['weights']
    opt.algorithm = best_params['algorithm']
    opt.n_neighbors = best_params['n_neighbors']"""

    opt.metric = 'manhattan'
    opt.weights = 'distance'
    opt.algorithm = 'ball_tree'
    opt.n_neighbors = 5

    if options.compare:
        opt.compare_metric()
        opt.compare_weights()
        opt.compare_n_neighbors()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()

    
if __name__ == "__main__":
    main()
