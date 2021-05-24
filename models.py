import metrics

import time

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder


class Models:
    def __init__(self, label_encoder: LabelEncoder,
                 feature_data: pd.DataFrame,
                 label_data: pd.DataFrame,
                 test_size: int,
                 app_metrics: metrics.Metrics,
                 grid_search: bool):
        self.labelEncoder = label_encoder
        self.featureData = feature_data
        self.labelData = label_data
        self.testSize = test_size
        self.appMetrics = app_metrics
        self.useGridSearch = grid_search

    def run_classifiers(self, entity_types: np.array):
        X_train, X_test, y_train, y_test = train_test_split(self.featureData, self.labelData,
                                                            test_size=self.testSize, random_state=1)

        self.labelEncoder.fit(entity_types)

        y_train = self.reshape_y_data(y_train)
        y_test = self.reshape_y_data(y_test)

        self.gaussian(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.svc(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.decision_tree(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.knn_nca(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.grid_search_rf(X_train, X_test, y_train[:, 0], y_test[:, 0])

    def gaussian(self, X_train, X_test, y_train, y_test):
        start = time.time()
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        self.appMetrics.gaussianPerf = time.time() - start
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nGaussian NB: {}'.format(score))
        self.appMetrics.gaussianScore = score

    def svc(self, X_train, X_test, y_train, y_test):
        start = time.time()
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train)
        self.appMetrics.svmPerf = time.time() - start
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nSVM: {}'.format(score))
        self.appMetrics.svmScore = score

    def decision_tree(self, X_train, X_test, y_train, y_test):
        start = time.time()
        clf = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=1))
        clf.fit(X_train, y_train)
        self.appMetrics.decisionTreePerf = time.time() - start
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nDecision tree: {}'.format(score))
        self.appMetrics.decisionTreeScore = score

    # see: https://scikit-learn.org/stable/modules/neighbors.html#id4
    def knn_nca(self, X_train, X_test, y_train, y_test):
        start = time.time()
        nca = NeighborhoodComponentsAnalysis(random_state=1)
        knn = KNeighborsClassifier(n_neighbors=3)
        nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
        nca_pipe.fit(X_train, y_train)
        self.appMetrics.nca_knnPerf = time.time() - start
        score = '{}%'.format(nca_pipe.score(X_test, y_test) * 100)
        print('\nKNN & NCA: {}'.format(score))
        self.appMetrics.nca_knnScore = score

    def grid_search_rf(self, X_train, X_test, y_train, y_test):
        print('\nRandom forest:')
        start = time.time()
        if self.useGridSearch:
            print('\nGridSearchCV:\n')
            parameters = [{'n_estimators': [500, 1000, 2000], 'criterion': ['gini', 'entropy'],
                           'max_features': ['sqrt', 'log2']}]
            grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters,
                                       scoring='accuracy', cv=10, n_jobs=-1, verbose=10)
            grid_search.fit(X_train, y_train)
            y_true, y_pred = y_test, grid_search.predict(X_test)
            out = '\n\nGrid search for Random Forest:\nExecution Time: {}\nUsedParameters {}\nBest params:\n{}\nBest score: {}\n***Classification Report:\n{}' \
                .format(time.time() - start, parameters, grid_search.best_params_, grid_search.best_score_,
                        classification_report(y_true, y_pred))
            print(out)
            self.appMetrics.gridSearchMetrics = out
            print('Executing Random Forest with best paramaters...')
            self.random_forest(X_train, X_test, y_train, y_test, grid_search.best_params_)
        else:
            params = {'n_estimators': 500, 'criterion': 'entropy', 'max_features': 'sqrt'}
            self.random_forest(X_train, X_test, y_train, y_test, params)

    def random_forest(self, X_train, X_test, y_train, y_test, params):
        start = time.time()
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(**params, n_jobs=-1, verbose=1))
        clf.fit(X_train, y_train)
        self.appMetrics.random_forestPerf = time.time() - start
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nRandom Forest: {}'.format(score))
        self.appMetrics.random_forestScore = score

    def reshape_y_data(self, y_data):
        return self.labelEncoder.transform(y_data).reshape(-1, 1)
