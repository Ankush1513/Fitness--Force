# src/train.py
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dataset import get_splits
from model import build_model

OUT_DIR = "/Users/ankushaggarwal/Desktop/fitness-pose-project/models"
os.makedirs(OUT_DIR, exist_ok=True)

def train(epochs=50, batch_size=32):
    X_train, y_train, X_val, y_val, X_test, y_test, le = get_splits()
    seq_len = X_train.shape[1]
    feat_dim = X_train.shape[2]
    n_classes = len(np.unique(y_train))
    model = build_model(seq_len, feat_dim, n_classes)
    print(model.summary())
    ckpt = ModelCheckpoint(os.path.join(OUT_DIR, "best_model.h5"), monitor="val_accuracy", save_best_only=True, verbose=1)
    es = EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[ckpt, es])
    # evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", acc)
    # save label encoder
    import joblib
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
    model.save(os.path.join(OUT_DIR, "final_model.h5"))

if __name__ == "__main__":
    train()
