import datetime
import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pydot

import os

class KerasTooling:
    def __init__(self, k_xyDf: pd.DataFrame, u_xyDf:pd.DataFrame, label_data: pd.DataFrame, entity_types):
        self.k_in, self.k_out = self.flatten_dataframe(k_xyDf)
        self.entity_types = numpy.delete(entity_types, 2)

        self.X_Train, self.X_Test, self.y_train, self.y_test = train_test_split(self.k_in, self.k_out, test_size=0.2, random_state=1)

    def flatten_dataframe(self, df):
        cols = ['Velocity', 'Altitude', 'Heading']
        X_df = df[cols]
        y_data = df['entityType'].to_numpy()
        dim0 = X_df.shape[0]
        dim1 = X_df['Velocity'].iloc[0].shape[0]
        dim2 = X_df.shape[1]
        X_data = np.empty(shape=(dim0, dim1, dim2))
        for index3, col_name in enumerate(cols):
            for index1 in range(dim0):
                X_data[index1, :, index3] = X_df[col_name].iloc[index1].to_numpy()
        return X_data, y_data

    def train(self, model):
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(np.array(self.entity_types).reshape(-1, 1))

        self.y_train = encoder.fit_transform(np.array(self.y_train).reshape(-1, 1))
        self.y_test = encoder.fit_transform(np.array(self.y_test).reshape(-1, 1))

        print(model.summary())

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='models/model_{epoch}',
                save_freq='epoch'
            ),
            keras.callbacks.TensorBoard(log_dir='logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        ]

        model.fit(self.X_Train, self.y_train, batch_size=17, epochs=20, validation_split=0.2, callbacks=callbacks, verbose=1)
        model.save('train')

        self.evaluate(model)

    def run_model(self):
        if os.path.exists('train'):
            print('Reloading existing model...\n')
            self.train(keras.models.load_model('train'))
        else:
            print('Creating model...\n')
            self.train(self.create_model())

    def create_model(self):
        inputs = keras.layers.Input(self.X_Train.shape[1:])
        lay1 = keras.layers.Conv1D(filters=256, kernel_size=3, padding="same", activation='relu')(inputs)
        lay1 = keras.layers.BatchNormalization()(lay1)
        lay2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same", activation='relu')(lay1)
        lay2 = keras.layers.BatchNormalization()(lay2)
        lay3 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same", activation='relu')(lay2)
        lay3 = keras.layers.BatchNormalization()(lay3)
        pooling = keras.layers.GlobalAveragePooling1D()(lay3)
        outputs = keras.layers.Dense(len(self.entity_types), activation="softmax")(pooling)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                           metrics=['categorical_crossentropy', 'categorical_accuracy', 'accuracy'])

        return model

    def evaluate(self, model):
        loss, acc, *catch = model.evaluate(self.X_Test, self.y_test, batch_size=17, verbose=1)
        print('\nLoss: {}\n'.format(loss))
        print('Accuracy: {}\n'.format(acc))

        pred = model.predict(self.X_Test, batch_size=17, verbose=1)
        print(pred)

    def run(self):
        self.run_model()