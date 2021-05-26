import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pydot

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

    def build_model(self):
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(np.array(self.entity_types).reshape(-1, 1))

        self.y_train = encoder.fit_transform(np.array(self.y_train).reshape(-1, 1))
        print(encoder.get_feature_names())
        print(self.y_train.shape)
        self.y_test = encoder.fit_transform(np.array(self.y_test).reshape(-1, 1))

        inputs = keras.layers.Input(self.X_Train.shape[1:])
        lay1 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(inputs)
        lay1 = keras.layers.BatchNormalization()(lay1)
        lay1 = keras.layers.ReLU()(lay1)
        lay2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(lay1)
        lay2 = keras.layers.BatchNormalization()(lay2)
        lay2 = keras.layers.ReLU()(lay2)
        lay3 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(lay2)
        lay3 = keras.layers.BatchNormalization()(lay3)
        lay3 = keras.layers.ReLU()(lay3)
        pooling = keras.layers.GlobalAveragePooling1D()(lay3)
        outputs = keras.layers.Dense(len(self.entity_types), activation="softmax")(pooling)

        model = keras.Model(inputs=inputs, outputs=outputs)
        print(model.summary())

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_crossentropy", "categorical_accuracy"])

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='models/model_{epoch}',
                save_freq='epoch'
            ),
            keras.callbacks.TensorBoard(log_dir='logs')
        ]

        model.fit(self.X_Train, self.y_train, batch_size=32, epochs=100, callbacks=callbacks, verbose=1)

    def evaluate(self, model):
        loss, acc = model.evaluate()

    def run(self):
        self.build_model()