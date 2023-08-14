# ++++ ENVIRONMENT SETUP ++++

import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from datagen import DataGenerator # Custom class

SAVE_PATH = "/path/to/save/" # Should end with /
MODEL_TO_USE = "mlp" # Should be 'mlp' or 'cnn'

EPOCHS = 200
PARAMS = {"batch_size": 32,
          "views": 3,
          "planes": 500,
          "cells": 500,
          "n_channels": 3}
LEARN_RATE = 0.00001

EARLY_STOP = callbacks.EarlyStopping(min_delta=0.001, patience=5, monitor="val_loss",
                                     verbose=1, restore_best_weights=True)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)

VAL_SPLIT = 0.2
FILES = [file[:-3] for file in glob("/path/to/batch/images.gz")]
np.random.shuffle(FILES)
TRAIN_FILES = FILES[int(VAL_SPLIT*len(FILES)):]
VAL_FILES = FILES[:int(VAL_SPLIT*len(FILES))]
TRAIN_GENERATOR = DataGenerator(TRAIN_FILES, **PARAMS)
VAL_GENERATOR = DataGenerator(VAL_FILES, **PARAMS)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# ++++ CREATE MODEL ++++

def build_model(planes, cells, channels, model_type) -> models.Sequential:
    'Creates a model suitable for NuEnergy regression.'
    if model_type == "mlp":
        model = models.Sequential([
            layers.InputLayer(input_shape=(planes, cells, channels)),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(1, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="softplus")
        ])
    elif model_type == "cnn":
        model = models.Sequential([
            layers.InputLayer(input_shape=(planes, cells, channels)),
            layers.Conv2D(32, (2,2), activation="relu"),
            layers.MaxPooling2D((2,2)),   
            layers.Conv2D(32, (2,2), activation="relu"),
            layers.MaxPooling2D((2,2)),   
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="softplus")
        ])
    else:
        print("Invalid model type! Please use either 'mlp' or 'cnn.'")
        return
    return model

model = build_model(PARAMS["planes"], PARAMS["cells"], PARAMS["n_channels"], MODEL_TO_USE)
model.summary()

model.compile(optimizer=OPTIMIZER, loss="mean_squared_error")
history = model.fit(TRAIN_GENERATOR, validation_data=VAL_GENERATOR,
                    epochs=EPOCHS, callbacks=EARLY_STOP)

model.save(SAVE_PATH)

# ++++ CREATE PLOTS ++++

plt.plot(history.history["loss"], label="Train. Loss")
plt.plot(history.history["val_loss"], label="Val. Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(loc="upper right")
plt.savefig(SAVE_PATH + "losses.png")