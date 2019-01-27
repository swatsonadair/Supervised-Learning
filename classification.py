import numpy as np
import matplotlib.pyplot as plt

import time

from load_datasets import *

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve

def pretty_print(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


class Classification(object):

    def __init__(self):
        pass


    def fit_and_predict(self):

        results = {}

        clf = self.create_classifier()
        clf.fit(self.X, self.y)
        y_pred = clf.predict(self.X)
        results['Training Score'] = f1_score(self.y, y_pred, average='macro')
        results['Training Accuracy'] = accuracy_score(self.y, y_pred)

        y_pred = clf.predict(self.X_test)
        results['Test Score'] = f1_score(self.y_test, y_pred, average='macro')
        results['Test Accuracy'] = accuracy_score(self.y_test, y_pred)

        pretty_print(results)

    def plot_learning_curve(self):

        clf = self.create_classifier()
        train_sizes, train_scores, test_scores = learning_curve(clf, self.X, self.y, cv=self.k_fold, scoring='f1_macro')

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.xlabel('Training Size')
        plt.ylabel('Score')

        # Save graph
        plt.grid()
        plt.title(self.title)
        plt.legend(loc="best")
        plt.savefig('output/' + self.method + '-' + self.dataset + '-learning-score.png', dpi=100, bbox_inches="tight")
        plt.close("all")


    def plot_learning_curve_time(self):

        print 'Learning Curve Time'
        results = {}
            
        for hyperparam in range(200, self.X.shape[0], self.X.shape[0]/24):

            results[hyperparam] = {}

            X = self.X[0:hyperparam]
            y = self.y[0:hyperparam]

            # Perform Training
            clf = self.create_classifier()

            start = time.time()
            clf.fit(X, y)
            train_end = time.time()
            train_elapsed = train_end - start
            results[hyperparam]['Fit Time'] = train_elapsed

            start = time.time()
            y_pred = clf.predict(X)
            predict_end = time.time()
            predict_elapsed = predict_end - start
            results[hyperparam]['Predict Time'] = predict_elapsed


        #pretty_print(results)
        self.plot_times(results, 'size', 'Training Size')


    def plot_categories(self, results, filename, xlabel):

        colors = ['b', 'g', 'm', 'y']
        i = 0

        #print results
        n_groups = len(results)
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.25

        x_axis = np.array(sorted(results.iterkeys()))
        data = []
        for x in x_axis:
            data.append(results[x])

        #trend = [d['Training Score'] for d in data]
        #plt.bar(index, trend, bar_width, color=colors[i], label='Training Score')
        #i += 1

        trend = [d['CV Training Score'] for d in data]
        plt.bar(index + bar_width, trend, bar_width, color=colors[i], label='Training Score')
        i += 1

        trend = [d['CV Test Score'] for d in data]
        plt.bar(index + bar_width * 2, trend, bar_width, color=colors[i], label='Cross-validation Score')
        i += 1

        # X-axis
        plt.xlabel(xlabel)
        plt.xticks(index + bar_width, x_axis)

        # Y-axis
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(.1, 1.05, step=0.1))
        
        # Add Parameter values text
        txt = ''
        for key, value in self.__dict__.iteritems():
            if key not in ['title', 'dataset', 'X', 'y', 'method', 'X_test', 'y_test']:
                txt += key + ': ' + str(value) + ', '
        plt.figtext(0.5, 0, txt, wrap=True, horizontalalignment='center', fontsize=8)

        # Save graph
        plt.grid()
        plt.title(self.title)
        plt.legend(loc="best", borderaxespad=0.)
        plt.savefig('output/' + self.method + '-' + self.dataset + '-' + filename + '-score.png', dpi=100, bbox_inches="tight")
        plt.close("all")


    def plot_score(self, results, filename, xlabel):

        colors = ['.b-', '.g-', '.m-', '.y-']
        index = 0

        x_axis = np.array(sorted(results.iterkeys()))
        data = []
        for x in x_axis:
            data.append(results[x])

        # REMOVE THESE LINES TO NOT DISPLAY
        #trend = [d['Training Score'] for d in data]
        #plt.plot(x_axis, trend, '.m-', label='Training Score (w/o CV)')

        trend = [d['CV Training Score'] for d in data]
        plt.plot(x_axis, trend, colors[index], label='Training Score')
        index += 1

        trend = [d['CV Test Score'] for d in data]
        plt.plot(x_axis, trend, colors[index], label='Cross-validation Score')
        index += 1

        # X-axis
        plt.xlabel(xlabel)

        # Y-axis
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(.1, 1.05, step=0.1))
        
        # Add Parameter values text
        txt = ''
        for key, value in self.__dict__.iteritems():
            if key not in ['title', 'dataset', 'X', 'y', 'method', 'X_test', 'y_test']:
                txt += key + ': ' + str(value) + ', '
        plt.figtext(0.5, 0, txt, wrap=True, horizontalalignment='center', fontsize=8)

        # Save graph
        plt.grid()
        plt.title(self.title)
        plt.legend(loc="best", borderaxespad=0.)
        plt.savefig('output/' + self.method + '-' + self.dataset + '-' + filename + '-score.png', dpi=100, bbox_inches="tight")
        plt.close("all")


    def plot_times(self, results, filename, xlabel):
        
        colors = ['.b-', '.g-', '.m-', '.y-']
        index = 0

        x_axis = np.array(sorted(results.iterkeys()))
        data = []
        for x in x_axis:
            data.append(results[x])

        trend = [d['Fit Time'] for d in data]
        plt.plot(x_axis, trend, colors[index], label='Fit Time')
        index += 1

        """trend = [d['Predict Time'] for d in data]
        plt.plot(x_axis, trend, colors[index], label='Predict Time')
        index += 1"""

         # X-axis
        plt.xlabel(xlabel)

        # Y-axis
        plt.ylabel('Time')
        
        # Add Parameter values text
        txt = ''
        for key, value in self.__dict__.iteritems():
            if key not in ['title', 'dataset', 'X', 'y', 'method', 'X_test', 'y_test']:
                txt += key + ': ' + str(value) + ', '
        plt.figtext(0.5, 0, txt, wrap=True, horizontalalignment='center', fontsize=8)

        # Save graph
        plt.grid()
        plt.title(self.title)
        plt.legend(loc="best", borderaxespad=0.)
        plt.savefig('output/' + self.method + '-' + self.dataset + '-' + filename + '-time.png', dpi=100, bbox_inches="tight")
        plt.close("all")

    def plot_depth(self, results, filename, xlabel):

        colors = ['.b-', '.g-', '.m-', '.y-']
        index = 0

        x_axis = np.array(sorted(results.iterkeys()))
        data = []
        for x in x_axis:
            data.append(results[x])

        trend = [d['Depth'] for d in data]
        plt.plot(x_axis, trend, colors[index], label='Depth')
        index += 1

         # X-axis
        plt.xlabel(xlabel)

        # Y-axis
        plt.ylabel('Depth')
        
        # Add Parameter values text
        txt = ''
        for key, value in self.__dict__.iteritems():
            if key not in ['title', 'dataset', 'X', 'y', 'method', 'X_test', 'y_test']:
                txt += key + ': ' + str(value) + ', '
        plt.figtext(0.5, 0, txt, wrap=True, horizontalalignment='center', fontsize=8)

        # Save graph
        plt.grid()
        plt.title(self.title)
        plt.legend(loc="best", borderaxespad=0.)
        plt.savefig('output/' + self.method + '-' + self.dataset + '-' + filename + '-depth.png', dpi=100, bbox_inches="tight")
        plt.close("all")
