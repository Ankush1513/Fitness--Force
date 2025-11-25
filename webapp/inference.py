# src/inference.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
from tensorflow.keras.models import load_model
from preprocess import compute_features_for_frame  # NEW — uses angles + normalized coords

mp_pose = mp.solutions.pose

MODEL_PATH = "/Users/ankushaggarwal/Desktop/fitness-pose-project/models/final_model.h5"
LE_PATH = "/Users/ankushaggarwal/Desktop/fitness-pose-project/models/label_encoder.pkl"

# Load the trained model + label encoder
model = load_model(MODEL_PATH)
le = joblib.load(LE_PATH)


def extract_feature_sequence_from_video(video_path):
    """Extract per-frame features (normalized xyz + angles) from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                # Convert landmarks to xyzw format required by preprocess
                lm = [
                    [l.x * w, l.y * h, l.z * w, l.visibility]
                    for l in results.pose_landmarks.landmark
                ]
                feats = compute_features_for_frame(lm)
            else:
                feats = compute_features_for_frame(None)

            frames.append(feats)

    cap.release()
    return np.array(frames)


def predict_on_video(video_path, seq_len=60):
    """Predict labels for sliding windows across the entire video."""
    feats = extract_feature_sequence_from_video(video_path)

    windows = []
    step = seq_len // 2  # 50% overlap

    for start in range(0, len(feats) - seq_len + 1, step):
        w = feats[start:start + seq_len]
        windows.append((start, w))

    preds = []

    for start, win in windows:
        inp = np.expand_dims(win, 0)  # Shape: (1, seq_len, features)
        p = model.predict(inp, verbose=0)[0]
        label_idx = np.argmax(p)
        label = le.inverse_transform([label_idx])[0]

        preds.append({
            "start": start,
            "label": label,
            "conf": float(p[label_idx])
        })

    return preds


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py video_file.mp4")
        exit()

    video_path = sys.argv[1]
    results = predict_on_video(video_path)

    for r in results:
        print(r)
