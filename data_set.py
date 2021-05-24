import models
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
        self.windowSize = args.window_size
        self.stepSize = args.step_size
        self.testSize = args.test_size
        self.cols = ['entityType', 'Velocity', 'Altitude', 'Heading', 'Roll', 'Pitch', 'Yaw']

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
                               'Latitude', 'U', 'V'], axis=1, inplace=True)
                self.xyDf = pd.concat([self.xyDf, dcs_data])  # merge temp df into parent df
                print('Processed {}'.format(x))
            self.app_metrics.xyPerf = time.time() - start  # pass time taken
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
                self.windowDf.at[index, 'Roll'] = subset['Roll'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Pitch'] = subset['Pitch'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Yaw'] = subset['Yaw'].iloc[i:i + self.windowSize]
                index += 1
        self.app_metrics.windowPerf = time.time() - start
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
            r = self.windowDf['Roll'].iloc[iWindow]
            p = self.windowDf['Pitch'].iloc[iWindow]
            y = self.windowDf['Yaw'].iloc[iWindow]
            dataList = [[iWindow, n, e, v.iloc[n], a.iloc[n], h.iloc[n], r.iloc[n], p.iloc[n], y.iloc[n]] for n in range(nSeriesSize)]
            flatWindowDFList.append(pd.DataFrame(dataList, columns=colLabels))
        self.flatWindowDF = pd.concat(flatWindowDFList)
        self.flatWindowDF.to_pickle(self.flatWindowPickle)

    def generate_features(self):
        start = time.time()
        self.flatWindowDF.astype({'windowID': int, 'timeID': int, 'entityType': str, 'Velocity': float, 'Altitude': float, 'Heading': float, 'Roll': float, 'Pitch': float, 'Yaw': float})
        xDataDF = self.flatWindowDF[['windowID', 'timeID', 'Velocity', 'Altitude', 'Heading', 'Roll', 'Pitch', 'Yaw']]
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

        ml_models = models.Models(self.labelEncoder, self.featureData, self.labelData, self.testSize, self.app_metrics,
                                    self.gridSearch)
        ml_models.run_classifiers(self.entityTypes)
