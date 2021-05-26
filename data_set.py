import sklearn_tooling
import keras_tooling
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

from sklearn.preprocessing import LabelEncoder


class DcsData:
    def __init__(self, args, app_metrics: metrics.Metrics):
        self.app_metrics = app_metrics

        self.subfolder = 'data/train/'
        self.data = args.data
        self.unknown = args.unknown
        self.gridSearch = args.grid_search
        self.rerun = args.rerun
        self.mlTool = args.ml_tool
        self.windowSize = args.window_size
        self.stepSize = args.step_size
        self.testSize = args.test_size
        self.cols = ['entityType', 'Velocity', 'Altitude', 'Heading']#, 'Roll', 'Pitch', 'Yaw']

        self.known_xyDf = pd.DataFrame(columns=self.cols)
        self.known_xyDfPickle = 'pickles/dataFrame_known.pkl'
        self.unknown_xyDf = pd.DataFrame(columns=self.cols)
        self.unknown_xyDfPickle = 'pickles/dataFrame_unknown.pkl'

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

        self.entityTypes = np.array([])

        self.labelEncoder = LabelEncoder()

        if not os.path.exists('pickles'):
            os.mkdir('pickles')
        self.SAMPLE_RATE: Final = 10

    def create_xy_df(self) -> None:
        print('Loading DCS data...\n')
        start = time.time()
        routes = ['Route1', 'Route2', 'Route3']
        datasets = []
        for r in routes:
            print('Processing {}'.format(r))
            with zipfile.ZipFile(self.subfolder + r + '_training.zip', 'r') as zip:
                files = []
                if not self.unknown:
                    files = [f for f in zip.namelist() if 'Unknown' not in f]
                else:
                    files = [f for f in zip.namelist()]
                entities = []
                for x in files:
                    dcs_data = pd.read_csv(zip.open(x))
                    dcs_data.drop(labels=range(dcs_data.shape[0]-501, dcs_data.shape[0]-1), axis=0, inplace=True)
                    if dcs_data['entityType'].iloc[0] not in entities:
                        entities.append(dcs_data['entityType'].iloc[0])

                    velocities = np.array([distance((dcs_data['Latitude'].iloc[i], dcs_data['Longitude'].iloc[i]),
                                           (dcs_data['Latitude'].iloc[i+1], dcs_data['Longitude'].iloc[i+1])).meters
                                            * self.SAMPLE_RATE
                                  for i in range(dcs_data.shape[0]-1)])
                    velocities = np.append(velocities, velocities[[-1]])
                    dcs_data['Velocity'] = velocities
                    dcs_data.drop(['timestamp', 'id', 'entityClass', 'Longitude',
                                   'Latitude', 'U', 'V', 'Roll', 'Pitch', 'Yaw'], axis=1, inplace=True)


                    print('Processed {}'.format(x))
                    datasets.append(dcs_data)
            print('Done {}'.format(r))
        xyDf = pd.concat(datasets, axis=0)  # datasets into parent
        self.known_xyDf = xyDf[xyDf['entityType'] != 'Unknown']  # separate knowns and unknowns
        self.unknown_xyDf = xyDf[xyDf['entityType'] == 'Unknown']
        self.app_metrics.xyPerf = time.time() - start  # pass time taken
        self.known_xyDf.to_pickle(self.known_xyDfPickle)
        self.unknown_xyDf.to_pickle(self.unknown_xyDfPickle)
        print('Wrote to pickle file: {} & {}\n'.format(self.known_xyDfPickle, self.unknown_xyDfPickle))
        self.entityTypes = np.array(entities)
        np.save(self.entityNpy, self.entityTypes)

    def create_windowed_df(self, dataframe, pickle):
        if not self.entityTypes.size > 0:
            print('Entity types not loaded, re-run DCS data pre-processing')
            exit(-1)
        print('Creating data frame using Sliding Window technique...')
        start = time.time()
        index = 0
        self.windowDf = pd.DataFrame(columns=self.cols)
        for entity in self.entityTypes:
            print('Processing {}'.format(entity))
            subset = dataframe.loc[dataframe['entityType'] == entity]  # retrieve all rows per aircraft
            indexed_list = list(range(0, len(subset) - self.windowSize, self.stepSize))
            for i in indexed_list:
                self.windowDf.at[index, 'entityType'] = subset['entityType'].iloc[i]
                self.windowDf.at[index, 'Velocity'] = subset['Velocity'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Altitude'] = subset['Altitude'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Heading'] = subset['Heading'].iloc[i:i + self.windowSize]
                #self.windowDf.at[index, 'Roll'] = subset['Roll'].iloc[i:i + self.windowSize]
                #self.windowDf.at[index, 'Pitch'] = subset['Pitch'].iloc[i:i + self.windowSize]
                #self.windowDf.at[index, 'Yaw'] = subset['Yaw'].iloc[i:i + self.windowSize]
                index += 1
        self.app_metrics.windowPerf = time.time() - start
        self.windowDf.to_pickle(pickle)
        print('Wrote to pickle file: {}\n'.format(pickle))
        return self.windowDf

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
            #r = self.windowDf['Roll'].iloc[iWindow]
            #p = self.windowDf['Pitch'].iloc[iWindow]
            #y = self.windowDf['Yaw'].iloc[iWindow]
            dataList = [[iWindow, n, e, v.iloc[n], a.iloc[n], h.iloc[n]] for n in range(nSeriesSize)]#, r.iloc[n], p.iloc[n], y.iloc[n]] for n in range(nSeriesSize)]
            flatWindowDFList.append(pd.DataFrame(dataList, columns=colLabels))
        self.flatWindowDF = pd.concat(flatWindowDFList)
        self.flatWindowDF.to_pickle(self.flatWindowPickle)

    def generate_features(self):
        start = time.time()
        self.flatWindowDF.astype({'windowID': int, 'timeID': int, 'entityType': str, 'Velocity': float, 'Altitude': float, 'Heading': float})#, 'Roll': float, 'Pitch': float, 'Yaw': float})
        xDataDF = self.flatWindowDF[['windowID', 'timeID', 'Velocity', 'Altitude', 'Heading']]#, 'Roll', 'Pitch', 'Yaw']]
        yDataDuplicateDF = self.flatWindowDF[['windowID', 'entityType']]
        extractedFeaturesDF = extract_features(xDataDF, column_id='windowID', column_sort="timeID", column_kind=None, column_value=None)
        impute(extractedFeaturesDF)
        self.labelData = (yDataDuplicateDF.drop_duplicates(subset='windowID'))['entityType']
        self.featureData = select_features(extractedFeaturesDF, self.labelData.to_numpy())
        self.app_metrics.featureExtractionPerf = time.time() - start
        self.featureData.to_pickle(self.featuresPickle)
        self.labelData.to_pickle(self.labelPickle)

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

        if (os.path.exists(self.known_xyDfPickle) and os.path.exists(self.unknown_xyDfPickle)) and not self.rerun:
            print('Reusing known and unknown pickle dataframes from {} and {}'.format(self.known_xyDfPickle, self.unknown_xyDfPickle))
            self.known_xyDf = pd.read_pickle(self.known_xyDfPickle)
            self.unknown_xyDf = pd.read_pickle(self.unknown_xyDfPickle)

        if self.mlTool == 'sklearn':
            sklearnTooling = sklearn_tooling.SklearnTooling(self.labelEncoder, self.featureData, self.labelData, self.testSize, self.app_metrics,
                                               self.gridSearch)
            sklearnTooling.run_classifiers(self.entityTypes)
        else:
            if os.path.exists('pickles/known_windowDf.pkl') and os.path.exists('pickles/unknown_windowDf.pkl'):
                self.known_xyDf = pd.read_pickle('pickles/known_windowDf.pkl')
                self.unknown_xyDf = pd.read_pickle('pickles/unknown_windowDf.pkl')
            else:
                self.known_xyDf = self.create_windowed_df(self.known_xyDf, 'pickles/known_windowDf.pkl')
                self.unknown_xyDf = self.create_windowed_df(self.unknown_xyDf, 'pickles/unknown_windowDf.pkl')
            kerasTooling = keras_tooling.KerasTooling(self.known_xyDf, self.unknown_xyDf, self.labelData, self.entityTypes)
            kerasTooling.run()


    def run_batch(self):
        print('Executing batch run to find most optimal GridSearchCV parameters for random tree...')
        self.xyDf = pd.read_pickle(self.xyDfPickle)
        self.entityTypes = np.load(self.entityNpy)
        self.windowDf = pd.read_pickle(self.windowDfPickle)
        self.flatWindowDF = pd.read_pickle(self.flatWindowPickle)
        self.labelData = pd.read_pickle(self.labelPickle)
        self.featureData = pd.read_pickle(self.featuresPickle)

        sklearnTooling = sklearn_tooling.SklearnTooling(self.labelEncoder, self.featureData, self.labelData,
                                                        self.testSize, self.app_metrics,
                                                        self.gridSearch)

        # initial values for baseline run
        params = [{'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy'],
                   'max_depth': [None, 2, 4, 6, 8, 10], 'min_samples_split': [2, 3, 4, 5],
                   'min_samples_leaf': [1, 2, 3, 4, 5], 'min_weight_fraction_leaf': [0.0, 0.2, 0.5, 0.8, 1.0, 2.0],
                   'max_features': ['sqrt', 'log2'], 'max_leaf_nodes': [None, 5, 10, 20, 50], 'min_impurity_decrease': [0.0, 0.2, 0.4, 0.6],
                   'bootstrap': [True, False], 'oob_score': [True, False], 'n_jobs': [None, 1, 2, 3, 4, 5, -1],
                   'random_state': [None, 1, 5, 10, 50], 'verbose': [10], 'warm_start': [True, False],
                   'class_weight': ['balanced', 'balanced_subsample'],
                   'ccp_alpha': [0.0, 0.2, 0.5, 1.0, 1.5], 'max_samples': [None, 50, 100, 500, 1000]}]

        times_to_optimise = 10
        for i in range(times_to_optimise):
            best_params, score = sklearnTooling.run_classifiers(params, self.entityTypes)
            print(best_params)