import os

from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D

import config
from hx.utils.plt_utils import plot_loss_acc
from hx.callbacks.epoch_checkpoint import EpochCheckpoint
from hx.callbacks.training_monitor import TrainingMonitor


class Autoencoder():
    def __init__(self):
        self.img_rows = config.img_row
        self.img_cols = config.img_col
        self.channels = config.channels
        self.img_shape = config.img_shape

        optimizer = Adam(lr=0.001)

        self.model = self.build_model()
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()

    def build_model(self):
        input_layer = Input(shape=self.img_shape)

        # encoder
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        h = MaxPooling2D((2, 2), padding='same')(h)

        # decoder
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
        h = UpSampling2D((2, 2))(h)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)

        return Model(input_layer, output_layer)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
        callbacks = [
            EpochCheckpoint(output_path=config.checkpoints, every=2, start_at=config.start_epoch),
            TrainingMonitor(config.fig_path, json_path=config.json_path, start_at=config.start_epoch),
            early_stopping
        ]
        H = self.model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(x_val, y_val),
                           callbacks=callbacks,
                           verbose=1)
        print(H)
        print(H.history)
    def eval_model(self, x_test):
        preds = self.model.predict(x_test)
        return preds

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        return load_model(filename)
