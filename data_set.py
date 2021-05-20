import metrics

import zipfile
import os
import time
from typing import Final

import numpy as np
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from geopy.distance import distance

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler



class DcsData:
    def __init__(self, args, app_metrics: metrics.Metrics):
        self.app_metrics = app_metrics

        self.subfolder = 'data/train/'
        self.data = args.data
        self.unknown = args.unknown
        self.rerun = args.rerun
        self.windowSize = args.window_size
        self.stepSize = args.step_size
        self.testSize = args.test_size
        self.cols = ['entityType', 'Velocity', 'Altitude', 'Heading']

        self.xyDf = pd.DataFrame(columns=self.cols)
        self.xyDfPickle = 'pickles/dataFrame_{}_{}.pkl'.format(self.windowSize, self.stepSize)

        self.windowDf = pd.DataFrame(columns=self.cols)
        self.windowDfPickle = 'pickles/windowSeries_{}_{}.pkl'.format(self.windowSize, self.stepSize)

        self.flatWindowDF = pd.DataFrame(columns=self.cols)
        self.flatWindowPickle = 'pickles/flatWindow_{}_{}.pkl'.format(self.windowSize, self.stepSize)

        self.featureData = pd.DataFrame()
        self.featuresPickle = 'pickles/xFeaturesFiltered_{}_{}.pkl'.format(self.windowSize, self.stepSize)
        self.labelData = pd.DataFrame()
        self.labelPickle = 'pickles/yLabels_{}_{}.pkl'.format(self.windowSize, self.stepSize)

        self.entityNpy = 'pickles/entities.npy'

        self.labelEncoder = LabelEncoder()

        if not os.path.exists('pickles'):
            os.mkdir('pickles')
        self.SAMPLE_RATE: Final = 10

    def create_xy_df(self) -> None:
        print('Loading DCS data...\n')
        start = time.time()
        with zipfile.ZipFile(self.subfolder + self.data + '_training.zip', 'r') as zip:
            files = []
            if not self.unknown:
                files = [f for f in zip.namelist() if 'Unknown' not in f]
            else:
                files = [f for f in zip.namelist()]
            entities = []
            for x in files:
                dcs_data = pd.read_csv(zip.open(x))
                if dcs_data['entityType'].iloc[0] not in entities:
                    entities.append(dcs_data['entityType'].iloc[0])

                velocities = np.array([distance((dcs_data['Latitude'].iloc[i], dcs_data['Longitude'].iloc[i]),
                                       (dcs_data['Latitude'].iloc[i+1], dcs_data['Longitude'].iloc[i+1])).meters
                                        * self.SAMPLE_RATE
                              for i in range(dcs_data.shape[0]-1)])
                velocities = np.append(velocities, velocities[[-1]])
                dcs_data['Velocity'] = velocities
                dcs_data.drop(['timestamp', 'id', 'entityClass', 'Longitude',
                               'Latitude', 'Roll', 'Pitch', 'Yaw', 'U', 'V'], axis=1, inplace=True)
                self.xyDf = pd.concat([self.xyDf, dcs_data])  # merge temp df into parent df
                print('Processed {}'.format(x))
            self.app_metrics.set_xy(time.time() - start)  # pass time taken
            self.xyDf.to_pickle(self.xyDfPickle)
            print('Wrote to pickle file: {}\n'.format(self.xyDfPickle))
            self.entityTypes = np.array(entities)
            np.save(self.entityNpy, self.entityTypes)

    def create_windowed_df(self):
        if not self.entityTypes.size > 0:
            print('Entity types not loaded, re-run DCS data pre-processing')
            exit(-1)
        print('Creating data frame using Sliding Window technique...')
        start = time.time()
        index = 0
        for entity in self.entityTypes:
            print('Processing {}'.format(entity))
            subset = self.xyDf.loc[self.xyDf['entityType'] == entity]  # retrieve all rows per aircraft
            indexed_list = list(range(0, len(subset) - self.windowSize, self.stepSize))
            for i in indexed_list:
                self.windowDf.at[index, 'entityType'] = subset['entityType'].iloc[i]
                self.windowDf.at[index, 'Velocity'] = subset['Velocity'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Altitude'] = subset['Altitude'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Heading'] = subset['Heading'].iloc[i:i + self.windowSize]
                index += 1
        self.app_metrics.set_window(time.time() - start)
        self.windowDf.to_pickle(self.windowDfPickle)
        print('Wrote to pickle file: {}\n'.format(self.windowDfPickle))

    def process_for_ts_fresh(self):
        # Keep a separate map of id to entityClass
        # Each window is an id
        # Each window time series needs to be a column of time 1:windowSize
        nWindows = self.windowDf.shape[0]
        nSeriesSize = self.windowDf['Velocity'].iloc[0].shape[0]
        nFlatWindow = self.windowDf.shape[0]*nSeriesSize
        iRow = 0
        colLabels = ['windowID', 'timeID'] + self.cols
        flatWindowDFList = []
        for iWindow in range(nWindows):
            e = self.windowDf['entityType'].iloc[iWindow]
            v = self.windowDf['Velocity'].iloc[iWindow]
            a = self.windowDf['Altitude'].iloc[iWindow]
            h = self.windowDf['Heading'].iloc[iWindow]
            dataList = [[iWindow, n, e, v.iloc[n], a.iloc[n], h.iloc[n]] for n in range(nSeriesSize)]
            flatWindowDFList.append(pd.DataFrame(dataList, columns=colLabels))
        self.flatWindowDF = pd.concat(flatWindowDFList)
        self.flatWindowDF.to_pickle(self.flatWindowPickle)

    def generate_features(self):
        start = time.time()
        self.flatWindowDF.astype({'windowID': int, 'timeID': int, 'entityType': str, 'Velocity': float, 'Altitude': float, 'Heading': float})
        xDataDF = self.flatWindowDF[['windowID', 'timeID', 'Velocity', 'Altitude', 'Heading']]
        yDataDuplicateDF = self.flatWindowDF[['windowID', 'entityType']]
        extractedFeaturesDF = extract_features(xDataDF, column_id='windowID', column_sort="timeID", column_kind=None, column_value=None)
        impute(extractedFeaturesDF)
        self.labelData = (yDataDuplicateDF.drop_duplicates(subset='windowID'))['entityType']
        self.featureData = select_features(extractedFeaturesDF, self.labelData.to_numpy())
        self.app_metrics.set_feature_extraction(time.time() - start)
        self.featureData.to_pickle(self.featuresPickle)
        self.labelData.to_pickle(self.labelPickle)

    def reshape_y_data(self, y_data):
        return self.labelEncoder.transform(y_data).reshape(-1, 1)

    def run_classifiers(self):
        X_train, X_test, y_train, y_test = train_test_split(self.featureData, self.labelData, test_size=self.testSize, random_state=1)

        self.labelEncoder.fit(self.entityTypes)

        y_train = self.reshape_y_data(y_train)
        y_test = self.reshape_y_data(y_test)

        self.gaussian(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.svc(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.decision_tree(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.knn_ncs(X_train, X_test, y_train[:, 0], y_test[:, 0])
        self.random_forest(X_train, X_test, y_train[:, 0], y_test[:, 0])

    def gaussian(self, X_train, X_test, y_train, y_test):
        start = time.time()
        clf = GaussianNB()
        clf.fit(X_train.to_numpy(), y_train)
        self.app_metrics.set_gaussian_time(time.time() - start)
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nGaussian NB: {}'.format(score))
        self.app_metrics.set_gaussian_score(score)

    def svc(self, X_train, X_test, y_train, y_test):
        start = time.time()
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train)
        self.app_metrics.set_svm_time(time.time() - start)
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nSVM: {}'.format(score))
        self.app_metrics.set_svm_score(score)

    def decision_tree(self, X_train, X_test, y_train, y_test):
        start = time.time()
        clf = DecisionTreeClassifier(random_state=1)
        clf.fit(X_train, y_train)
        self.app_metrics.set_decision_tree_time(time.time() - start)
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nDecision tree: {}'.format(score))
        self.app_metrics.set_decision_tree_score(score)

    # see: https://scikit-learn.org/stable/modules/neighbors.html#id4
    def knn_ncs(self, X_train, X_test, y_train, y_test):
        start = time.time()
        nca = NeighborhoodComponentsAnalysis(random_state=1)
        knn = KNeighborsClassifier(n_neighbors=3)
        nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
        nca_pipe.fit(X_train, y_train)
        self.app_metrics.set_nca_knn_time(time.time() - start)
        score = '{}%'.format(nca_pipe.score(X_test, y_test) * 100)
        print('\nKNN & NCS: {}'.format(score))
        self.app_metrics.set_nca_knn_score(score)

    def random_forest(self, X_train, X_test, y_train, y_test):
        start = time.time()
        clf = RandomForestClassifier(n_estimators=100, random_state=1)
        clf.fit(X_train, y_train)
        score = '{}%'.format(clf.score(X_test, y_test) * 100)
        print('\nRandom Forest: {}'.format(score))

    def run(self):
        if (os.path.exists(self.xyDfPickle) and os.path.exists(self.entityNpy)) and not self.rerun:
            print('Reusing previous DCS dataframe pickle from {}'.format(self.xyDfPickle))
            self.xyDf = pd.read_pickle(self.xyDfPickle)
            self.entityTypes = np.load(self.entityNpy)
        else:
            self.create_xy_df()

        if os.path.exists(self.windowDfPickle) and not self.rerun:
            print('Reusing previous Sliding window pickle from {}'.format(self.windowDfPickle))
            self.windowDf = pd.read_pickle(self.windowDfPickle)
        else:
            self.create_windowed_df()

        if os.path.exists(self.flatWindowPickle) and not self.rerun:
            print('Reusing previous Flat window pickle from {}'.format(self.flatWindowPickle))
            self.flatWindowDF = pd.read_pickle(self.flatWindowPickle)
        else:
            self.process_for_ts_fresh()

        if (os.path.exists(self.labelPickle) and os.path.exists(self.featuresPickle)) and not self.rerun:
            print('Reusing previous Feature and Label pickle from {} and {}'.format(self.featuresPickle, self.labelPickle))
            self.labelData = pd.read_pickle(self.labelPickle)
            self.featureData = pd.read_pickle(self.featuresPickle)
        else:
            self.generate_features()

        self.run_classifiers()