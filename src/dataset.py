# src/dataset.py
import os, numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

PROC_DIR = "/Users/ankushaggarwal/Desktop/fitness-pose-project/data/processed"

def load_all():
    Xs, ys = [], []
    files = glob(os.path.join(PROC_DIR, "*_X.npy"))
    for fx in files:
        base = fx[:-6]  # remove _X.npy
        yfile = base + "_y.npy"
        if not os.path.exists(yfile):
            continue
        x = np.load(fx, allow_pickle=True)
        y = np.load(yfile, allow_pickle=True)
        Xs.append(x)
        ys.append(y)
    if not Xs:
        raise RuntimeError("No processed data found. Run preprocess.")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le

def get_splits(test_size=0.2, val_size=0.1):
    X, y, le = load_all()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    # from train, take validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), stratify=y_train)
    return X_train, y_train, X_val, y_val, X_test, y_test, le
