import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

from load_datasets import *
from classification import *

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn import svm


class SVCOptimizer(Classification):

    def __init__(self):
        self.method = 'svm'
        self.kernel = 'rbf'
        self.C = 1.0
        self.degree = 3
        self.gamma = 'auto'
        self.max_iter = -1
        self.class_weight = None
        self.k_fold = 10

    def create_classifier(self, kernel=None, degree=None, gamma=None, max_iter=None, class_weight=None):

        kernel = kernel if kernel is not None else self.kernel
        degree = degree if degree is not None else self.degree
        gamma = gamma if gamma is not None else self.gamma
        max_iter = max_iter if max_iter is not None else self.max_iter
        class_weight = class_weight if class_weight is not None else self.class_weight

        clf = svm.SVC(
                kernel=kernel, 
                degree=degree, 
                gamma=gamma, 
                max_iter=max_iter,
                class_weight=class_weight)

        return clf


    def get_support_vectors(self):

        print 'Compare vectors'
        results = {}

        clf = self.create_classifier()
        clf.fit(self.X, self.y)

        results['all'] = self.X.shape
        results['n_support'] = np.sum(clf.n_support_)

        pretty_print(results)


    def compare_kernel(self):

        print 'Compare kernel'
        results = {
            'linear': {},
            'rbf': {},
            'poly': {}
        }

        for hyperparam in results:
            
            # Perform Training
            clf = self.create_classifier(kernel=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(kernel=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'kernel', 'Kernel')

    def compare_degree(self): #Polynomial Only

        print 'Compare Degree'
        results = {}
        for i in range(0, 8):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(degree=hyperparam, kernel='poly')
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(degree=hyperparam, kernel='poly')
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])


        #pretty_print(results)
        self.plot_score(results, 'degree', 'Degree')
        self.plot_times(results, 'degree', 'Degree')


    def compare_gamma(self): #RBF, Poly, Sigmoid

        print 'Compare Gamma'
        results = {}
        for i in range(1, 9):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            gamma = hyperparam * 0.01

            # Perform Training
            clf = self.create_classifier(gamma=gamma)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(gamma=gamma)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'gm', 'Gamma')
        self.plot_times(results, 'gm', 'Gamma')


    def compare_weights(self):

        print 'Compare Weights'
        results = {
            'balanced': {},
            'None': {}
        }

        for hyperparam in results:
            print hyperparam
            
            class_weight = hyperparam
            if class_weight == 'None':
                class_weight = None

            # Perform Training
            clf = self.create_classifier(class_weight=class_weight)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(class_weight=class_weight)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'weights', 'Weights')


    def compare_max_iter(self):

        print 'Compare Max Iter'
        results = {}
        for i in range(1, 20):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            param_arg = hyperparam
            if param_arg == 19:
                param_arg = -1

            # Perform Training
            clf = self.create_classifier(max_iter=param_arg)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(max_iter=param_arg)
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

    opt = SVCOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'a'
    opt.title = 'SVM Classifier: Abalone'

    """params = {
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': np.arange(2, 8),
        'class_weight': ['balanced', None],
        'max_iter': np.arange(-1, 202, 20),
    }
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=params, cv=10, scoring='f1_macro', return_train_score=True)
    clf.fit(X, y)

    best_score = clf.best_score_
    print best_score

    best_params = clf.best_params_
    print best_params
    
    opt.kernel = best_params['kernel']
    opt.degree = best_params['degree']
    opt.class_weight = best_params['class_weight']
    opt.max_iter = best_params['max_iter']"""

    opt.kernel = 'linear'
    opt.class_weight = 'balanced'
    opt.max_iter = -1

    if options.compare:
        opt.get_support_vectors()
        opt.compare_kernel()
        opt.compare_degree()
        opt.compare_gamma()
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

    opt = SVCOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'l'
    opt.title = 'SVM Classifier: Letters'
    
    opt.kernel = 'rbf'
    opt.gamma = .06
    opt.class_weight = 'balanced'
    opt.max_iter = -1

    if options.compare:
        opt.get_support_vectors()
        opt.compare_kernel()
        opt.compare_degree()
        opt.compare_gamma()
        opt.compare_max_iter()
    
    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()
    
if __name__ == "__main__":
    main()
