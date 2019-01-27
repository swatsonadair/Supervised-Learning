import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

from load_datasets import *
from classification import *

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier


class NeuralNetworkOptimizer(Classification):

    def __init__(self):
        self.method = 'nn'
        self.solver = 'adam'
        self.activation = 'relu'
        self.max_iter = 200
        self.hidden_layer_sizes = (100,)
        self.k_fold = 10

    def create_classifier(self, solver=None, activation=None, max_iter=None, hidden_layer_sizes=None):

        solver = solver if solver is not None else self.solver
        activation = activation if activation is not None else self.activation
        max_iter = max_iter if max_iter is not None else self.max_iter
        hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes is not None else self.hidden_layer_sizes
        
        clf = MLPClassifier(
                solver=solver, 
                activation=activation, 
                max_iter=max_iter)

        return clf


    def get_layers(self):
        # Perform Training
        results = {}
        clf = self.create_classifier()
        clf.fit(self.X, self.y)
        y_pred = clf.predict(self.X)
        results['Training Score'] = f1_score(self.y, y_pred, average='macro')

        results['Classes'] = clf.classes_
        results['N Iter'] = clf.n_iter_
        results['Layers'] = clf.n_layers_
        results['Coefs'] = clf.coefs_

        for coef in clf.coefs_:
            print coef.shape

        pretty_print(results)

    def compare_solver(self):

        print 'Compare solver'
        results = {
            'lbfgs': {},
            'adam': {},
            'sgd': {}
        }

        for hyperparam in results:
            print hyperparam
            
            # Perform Training
            clf = self.create_classifier(solver=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(solver=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'solver', 'Solver')


    def compare_activation(self):

        print 'Compare activation'
        results = {
            'identity': {},
            'logistic': {},
            'tanh': {},
            'relu': {}
        }

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(activation=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(activation=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'act', 'Activation')

    def compare_layers(self):

        print 'Compare layers'
    
        results = {}
        for i in range(2, 8, 1):
        #for i in range(200, 2000, 100):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            hidden_layer_sizes = [100]*hyperparam

            # Perform Training
            clf = self.create_classifier(hidden_layer_sizes=hidden_layer_sizes)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(hidden_layer_sizes=hidden_layer_sizes)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'layer', 'Layers')
        self.plot_times(results, 'layer', 'Layers')


    def compare_neurons(self):

        print 'Compare neurons'
    
        results = {}
        for i in range(10, 150, 10):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            hidden_layer_sizes = (hyperparam,)

            # Perform Training
            clf = self.create_classifier(hidden_layer_sizes=hidden_layer_sizes)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(hidden_layer_sizes=hidden_layer_sizes)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'neurons', 'Neurons')
        self.plot_times(results, 'neurons', 'Neurons')


    def compare_max_iter(self):

        print 'Compare Max Iterations'
        results = {}
        if self.dataset == 'a':
            for i in range(20, 420, 40):
                results[i] = {}
        else:
            for i in range(20, 420, 40):
            #for i in range(200, 2000, 100):
                results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(max_iter=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(max_iter=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'iter', 'Max Iterations')
        self.plot_times(results, 'iter', 'Max Iterations')

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

    opt = NeuralNetworkOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'a'
    opt.title = 'Neural Network Classifier: Abalone'

    """params = {
        'solver': ['lbfgs', 'adam', 'sgd'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'max_iter': np.arange(20, 420, 40)
    }
    clf = GridSearchCV(estimator=MLPClassifier(), param_grid=params, cv=10, scoring='f1_macro', return_train_score=True)
    clf.fit(X, y)

    best_score = clf.best_score_
    print best_score

    best_params = clf.best_params_
    print best_params
    
    opt.solver = best_params['solver']
    opt.activation = best_params['activation']
    opt.max_iter = best_params['max_iter'])"""

    opt.solver = 'lbfgs'
    opt.activation = 'relu'
    opt.max_iter = 140

    if options.compare:
        opt.compare_layers()
        opt.compare_neurons()
        opt.compare_solver()
        opt.compare_activation()
        opt.compare_max_iter()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()


    #------- LETTERS ------#

    print 'Letters'

    X, y = load_letters()
    labels = np.array(np.unique(y))
    y = np.array([list(labels).index(v) for v in y])
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    opt = NeuralNetworkOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'l'
    opt.title = 'Neural Network Classifier: Letters'
    
    opt.solver = 'adam'
    opt.activation = 'relu'
    opt.max_iter = 160

    if options.compare:
        opt.compare_layers()
        opt.compare_neurons()
        opt.compare_solver()
        opt.compare_activation()
        opt.compare_max_iter()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()

    
if __name__ == "__main__":
    main()
