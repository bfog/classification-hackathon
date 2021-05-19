import zipfile
import os
from typing import Final

import numpy as np
import pandas as pd

from geopy.distance import distance

#from sklearn import preprocessing, svm
#from sklearn.preprocessing import LabelEncoder as le
#from sklearn.model_selection import train_test_split


class DcsData:
    def __init__(self, args):
        self.subfolder = 'data/train/'
        self.data = args.data
        self.unknown = args.unknown
        self.rerun = args.rerun

        self.xyDf = pd.DataFrame(columns=['entityType', 'Velocity', 'Altitude', 'Heading'])
        self.xyDfPickle = 'pickles/dataFrame.pkl'
        if not os.path.exists('pickles'):
            os.mkdir('pickles')
        self.SAMPLE_RATE: Final = 10

    def parse_data(self) -> None:
        print('Loading DCS data...\n')
        with zipfile.ZipFile(self.subfolder + self.data + '_training.zip', 'r') as zip:
            files = []
            if not self.unknown:
                files = [f for f in zip.namelist() if 'Unknown' not in f]
            else:
                files = [f for f in zip.namelist()]
            for x in files:
                dcs_data = pd.read_csv(zip.open(x))
                velocities = np.array([distance((dcs_data['Latitude'].iloc[i], dcs_data['Longitude'].iloc[i]),
                                       (dcs_data['Latitude'].iloc[i+1], dcs_data['Longitude'].iloc[i+1])).meters
                                        * self.SAMPLE_RATE
                              for i in range(dcs_data.shape[0]-1)])
                velocities = np.append(velocities, velocities[[-1]])
                dcs_data['Velocity'] = velocities
                dcs_data.drop(['timestamp', 'id', 'entityClass', 'Longitude',
                               'Latitude', 'Roll', 'Pitch', 'Yaw', 'U', 'V'], axis=1, inplace=True)
                self.xyDf = pd.concat([self.xyDf, dcs_data])
                print('Processed {}'.format(x))
            self.xyDf.to_pickle(self.xyDfPickle)
            print('Wrote to pickle file: {}'.format(self.xyDfPickle))

    # def execute(self):
    #     self.X = preprocessing.scale(self.X)
    #     X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
    #
    #     clf = svm.SVR()
    #     clf.fit(X_train, y_train)
    #     accuracy = clf.score(X_test, y_test)
    #     print(accuracy)

    def run(self):
        if os.path.exists(self.xyDfPickle) and not self.rerun:
            print('Reusing previous DCS dataframe pickle from {}'.format(self.xyDfPickle))
            self.xyDf = pd.read_pickle(self.xyDfPickle)
        else:
            self.parse_data()
