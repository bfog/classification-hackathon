import metrics

import zipfile
import os
import time
from typing import Final

import numpy as np
import pandas as pd

from geopy.distance import distance

#from sklearn import preprocessing, svm
#from sklearn.preprocessing import LabelEncoder as le
#from sklearn.model_selection import train_test_split


class DcsData:
    def __init__(self, args, app_metrics: metrics.Metrics):
        self.app_metrics = app_metrics

        self.subfolder = 'data/train/'
        self.data = args.data
        self.unknown = args.unknown
        self.rerun = args.rerun
        self.windowSize = args.window_size
        self.stepSize = args.step_size
        self.cols = ['entityType', 'Velocity', 'Altitude', 'Heading']

        self.xyDf = pd.DataFrame(columns=self.cols)
        self.xyDfPickle = 'pickles/dataFrame_{}_{}.pkl'.format(self.windowSize, self.stepSize)
        self.windowDf = pd.DataFrame(columns=self.cols)
        self.windowDfPickle = 'pickles/windowSeries_{}_{}.pkl'.format(self.windowSize, self.stepSize)
        self.entityNpy = 'pickles/entities.npy'

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
        for entity in self.entityTypes:
            subset = self.xyDf.loc[self.xyDf['entityType'] == entity]  # retrieve all rows per aircraft
            indexed_list = list(range(0, len(subset) - self.windowSize, self.stepSize))
            index = 0
            for i in indexed_list:
                self.windowDf.at[index, 'entityType'] = subset['entityType'].iloc[i]
                self.windowDf.at[index, 'Velocity'] = subset['Velocity'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Altitude'] = subset['Altitude'].iloc[i:i + self.windowSize]
                self.windowDf.at[index, 'Heading'] = subset['Heading'].iloc[i:i + self.windowSize]
                index += 1
        self.app_metrics.set_window(time.time() - start)
        self.windowDf.to_pickle(self.windowDfPickle)
        print('Wrote to pickle file: {}\n'.format(self.windowDfPickle))


    # def execute(self):
    #     self.X = preprocessing.scale(self.X)
    #     X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
    #
    #     clf = svm.SVR()
    #     clf.fit(X_train, y_train)
    #     accuracy = clf.score(X_test, y_test)
    #     print(accuracy)

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
