# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(seq_len, feature_dim, n_classes, dropout=0.3):
    inp = layers.Input((seq_len, feature_dim))
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
