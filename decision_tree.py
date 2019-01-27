import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

from load_datasets import *
from classification import *

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn import tree


class DecisionTreeOptimizer(Classification):

    def __init__(self):
        self.method = 'tree'
        self.criterion = 'gini'
        self.class_weight = None
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_features = None
        self.k_fold = 10


    def create_classifier(self, criterion=None, class_weight=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, max_features=None):

        criterion = criterion if criterion is not None else self.criterion
        class_weight = class_weight if class_weight is not None else self.class_weight
        max_depth = max_depth if max_depth is not None else self.max_depth
        min_samples_split = min_samples_split if min_samples_split is not None else self.min_samples_split
        min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else self.min_samples_leaf
        max_features = max_features if max_features is not None else self.max_features

        clf = tree.DecisionTreeClassifier(
                criterion=criterion, 
                class_weight=class_weight, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=1)

        return clf


    def compare_criterion(self):

        print 'Compare Criterion'
        results = {
            'entropy': {},
            'gini': {}
        }

        for hyperparam in results:
            print hyperparam
            
            # Perform Training
            clf = self.create_classifier(criterion=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')
            results[hyperparam]['Depth'] = clf.tree_.max_depth

            # Perform K-fold cross validation
            clf = self.create_classifier(criterion=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'crit', 'Criterion')


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
            results[hyperparam]['Depth'] = clf.tree_.max_depth

            # Perform K-fold cross validation
            clf = self.create_classifier(class_weight=class_weight)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'weights', 'Weights')


    def compare_max_depth(self):

        print 'Compare Depth'
        results = {}
        for i in range(1, 33):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(max_depth=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')
            results[hyperparam]['Depth'] = clf.tree_.max_depth

            # Perform K-fold cross validation
            clf = self.create_classifier(max_depth=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'depth', 'Max Depth')
        self.plot_times(results, 'depth', 'Max Depth')
        self.plot_depth(results, 'depth', 'Max Depth')


    def compare_min_samples_split(self):

        print 'Compare Min Samples Split'
        results = {}
        for i in range(2, 25):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(min_samples_split=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(min_samples_split=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'split', 'Min Samples Split')
        self.plot_times(results, 'split', 'Min Samples Split')


    def compare_min_samples_leaf(self):

        print 'Compare Min Samples Leaf'
        results = {}
        for i in range(1, 16):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(min_samples_leaf=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(min_samples_leaf=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'leaf', 'Min Samples Leaf')
        self.plot_times(results, 'leaf', 'Min Samples Leaf')


    def compare_max_features(self):

        print 'Compare Max Features'
        results = {}
        for i in range(1, self.X.shape[1] + 1):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(max_features=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(max_features=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'feat', 'Max Features')
        self.plot_times(results, 'feat', 'Max Features')


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
   
    opt = DecisionTreeOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'a'
    opt.title = 'Decision Tree Classifier: Abalone'
    
    """params = {
        'criterion': ['gini','entropy'],
        'max_depth': np.arange(3, 6, 1),
        'class_weight': ['balanced', None],
        'min_samples_split': np.arange(2, 25, 4),
        'min_samples_leaf': np.arange(1, 16, 4),
        'max_features': np.arange(1, X.shape[1] + 1, 2)
    }
    clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=params, cv=10, scoring='f1_macro', return_train_score=True)
    clf.fit(X, y)

    best_score = clf.best_score_
    print best_score

    best_params = clf.best_params_
    print best_params
    
    opt.criterion = best_params['criterion']
    opt.max_depth = best_params['max_depth']
    opt.class_weight = best_params['class_weight']
    opt.min_samples_split = best_params['min_samples_split']
    opt.min_samples_leaf = best_params['min_samples_leaf']
    opt.max_features = best_params['max_features']"""

    opt.criterion = 'entropy'
    opt.max_depth = 7
    opt.class_weight = None
    opt.min_samples_split = 12
    opt.min_samples_leaf = 5

    if options.compare:
        opt.compare_criterion()
        opt.compare_weights()
        opt.compare_max_depth()
        opt.compare_min_samples_split()
        opt.compare_min_samples_leaf()
        opt.compare_max_features()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()


    #------- LETTERS ------#

    print 'Letters'

    X, y = load_letters()
    labels = np.array(np.unique(y))
    y = np.array([list(labels).index(v) for v in y])

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    opt = DecisionTreeOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'l'
    opt.title = 'Decision Tree Classifier: Letters'

    """params = {
        'criterion': ['gini','entropy'],
        'max_depth': np.arange(3, 16, 3),
        'class_weight': ['balanced', None],
        'min_samples_split': np.arange(2, 25, 4),
        'min_samples_leaf': np.arange(1, 16, 4),
        'max_features': np.arange(1, X.shape[1] + 1, 2)
    }
    clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=params, cv=10, scoring='f1_macro', return_train_score=True)
    clf.fit(X, y)

    best_score = clf.best_score_
    print best_score

    best_params = clf.best_params_
    print best_params
    
    opt.criterion = best_params['criterion']
    opt.max_depth = best_params['max_depth']
    opt.class_weight = best_params['class_weight']
    opt.min_samples_split = best_params['min_samples_split']
    opt.min_samples_leaf = best_params['min_samples_leaf']
    opt.max_features = best_params['max_features']"""

    opt.criterion = 'entropy'
    opt.max_depth = 12
    opt.class_weight = None

    if options.compare:
        opt.compare_criterion()
        opt.compare_weights()
        opt.compare_max_depth()
        opt.compare_min_samples_split()
        opt.compare_min_samples_leaf()
        opt.compare_max_features()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()

    
if __name__ == "__main__":
    main()
