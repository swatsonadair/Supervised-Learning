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
from sklearn.ensemble import AdaBoostClassifier


class BoostingOptimizer(Classification):

    def __init__(self):
        self.method = 'boost'
        self.criterion = 'gini'
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_features = None
        self.class_weight = None
        self.k_fold = 10

        self.algorithm = 'SAMME'
        self.n_estimators = 50
        self.learning_rate = 1.


    def create_classifier(self, criterion=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, max_features=None, class_weight=None, algorithm=None, n_estimators=None, learning_rate=None):

        criterion = criterion if criterion is not None else self.criterion
        max_depth = max_depth if max_depth is not None else self.max_depth
        min_samples_split = min_samples_split if min_samples_split is not None else self.min_samples_split
        min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else self.min_samples_leaf
        max_features = max_features if max_features is not None else self.max_features
        class_weight = class_weight if class_weight is not None else self.class_weight

        algorithm = algorithm if algorithm is not None else self.algorithm
        n_estimators = n_estimators if n_estimators is not None else self.n_estimators
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate

        clf_tree = tree.DecisionTreeClassifier(
                criterion=criterion, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features, 
                class_weight=class_weight,
                random_state=1)

        clf = AdaBoostClassifier(base_estimator=clf_tree, algorithm=algorithm, n_estimators=n_estimators, learning_rate=learning_rate)

        return clf


    def compare_algorithm(self):

        print 'Compare Algorithm'
        results = {
            'SAMME': {},
            'SAMME.R': {}
        }

        for hyperparam in results:
            print hyperparam
            
            # Perform Training
            clf = self.create_classifier(algorithm=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(algorithm=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_categories(results, 'alg', 'Algorithm')


    def compare_n_estimators(self):

        print 'Compare N-Estimators'
        results = {}
        for i in range(5, 60, 5):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(n_estimators=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(n_estimators=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'n', 'N-Estimators')
        self.plot_times(results, 'n', 'N-Estimators')


    def compare_max_depth(self):

        print 'Compare Depth'
        results = {}
        for i in range(1, 21):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(max_depth=hyperparam)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(max_depth=hyperparam)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'depth', 'Max Depth')
        self.plot_times(results, 'depth', 'Max Depth')


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


    def compare_learning_rate(self):

        print 'Compare Learning Rate'
        results = {}
        for i in range(1, 5):
            results[i] = {}

        for hyperparam in results:
            print hyperparam

            # Perform Training
            clf = self.create_classifier(learning_rate=hyperparam * 0.2)
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            results[hyperparam]['Training Score'] = f1_score(self.y, y_pred, average='macro')

            # Perform K-fold cross validation
            clf = self.create_classifier(learning_rate=hyperparam * 0.2)
            scores = cross_validate(estimator=clf, X=self.X, y=self.y, cv=self.k_fold, scoring='f1_macro', return_train_score=True)
            results[hyperparam]['CV Training Score'] = np.mean(scores['train_score'])
            results[hyperparam]['CV Test Score'] = np.mean(scores['test_score'])
            results[hyperparam]['Fit Time'] = np.mean(scores['fit_time'])

        #pretty_print(results)
        self.plot_score(results, 'lr', 'Learning Rate')
        self.plot_times(results, 'lr', 'Learning Rate')


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

    opt = BoostingOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'a'
    opt.title = 'Boosting: Abalone'

    opt.criterion = 'entropy'
    opt.class_weight = None
    opt.min_samples_split = 12

    opt.algorithm = 'SAMME'
    opt.n_estimators = 30
    opt.max_depth = 5
    opt.min_samples_leaf = 5
    opt.learning_rate = 0.6

    if options.compare:
        opt.compare_algorithm()
        opt.compare_n_estimators()
        opt.compare_max_depth()
        opt.compare_min_samples_leaf()
        opt.compare_learning_rate()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()


    #------- LETTERS ------#

    print 'Letters'

    X, y = load_letters()
    labels = np.array(np.unique(y))
    y = np.array([list(labels).index(v) for v in y])

    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)

    opt = BoostingOptimizer()
    opt.X = X
    opt.y = y
    opt.X_test = X_test
    opt.y_test = y_test
    opt.dataset = 'l'
    opt.title = 'Boosting: Letters'

    opt.criterion = 'entropy'
    opt.class_weight = None

    opt.algorithm = 'SAMME'
    opt.n_estimators = 85
    opt.max_depth = 10
    opt.learning_rate = 0.8

    if options.compare:
        opt.compare_algorithm()
        opt.compare_n_estimators()
        opt.compare_max_depth()
        opt.compare_min_samples_leaf()
        opt.compare_learning_rate()

    opt.plot_learning_curve()
    opt.plot_learning_curve_time()
    opt.fit_and_predict()
    
    
if __name__ == "__main__":
    main()
