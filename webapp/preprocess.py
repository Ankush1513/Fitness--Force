# src/preprocess.py
import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm

RAW_DIR = "/Users/ankushaggarwal/Desktop/fitness-pose-project/data/raw"
PROC_DIR = "/Users/ankushaggarwal/Desktop/fitness-pose-project/data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

NUM_LM = 33   # Mediapipe pose = 33 keypoints
FEATURES_PER_LM = 3  # x, y, z (visibility is dropped)

# --- Landmark name → index mapping ---
LMAP = {name: idx for idx, name in enumerate([
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE","RIGHT_EYE_OUTER",
    "LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB",
    "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL","RIGHT_HEEL",
    "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
])}

def mp_index(name):
    return LMAP[name]


# ---------------- Geometry Helper ----------------

def angle_between(a, b, c):
    """Return angle (radians) at point b for triangle a-b-c."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom
    return np.arccos(np.clip(cosang, -1.0, 1.0))


# ---------------- Normalization -------------------

def normalize_landmarks_simple(landmarks):
    """
    Normalize coordinates by:
    1. Translating - remove hip center offset
    2. Scaling - divide by shoulder width
    """
    arr = np.array(landmarks)[:, :3]  # drop visibility

    # origin = midpoint of hips
    left_hip = arr[mp_index("LEFT_HIP")]
    right_hip = arr[mp_index("RIGHT_HIP")]
    origin = (left_hip + right_hip) / 2.0
    arr = arr - origin

    # scale = shoulder width
    left_sh = arr[mp_index("LEFT_SHOULDER")]
    right_sh = arr[mp_index("RIGHT_SHOULDER")]
    torso_width = np.linalg.norm(left_sh - right_sh) + 1e-6
    arr = arr / torso_width

    return arr.flatten()


# ---------------- Angle Feature Extraction -------------------

def compute_angle_features(landmarks):
    """
    Compute 8 meaningful biomechanical angles:
    Left/right: elbow, shoulder, knee, hip
    """
    arr = np.array(landmarks)[:, :3]
    angles = []

    # Elbow angles
    angles.append(angle_between(arr[mp_index("LEFT_SHOULDER")], arr[mp_index("LEFT_ELBOW")], arr[mp_index("LEFT_WRIST")]))
    angles.append(angle_between(arr[mp_index("RIGHT_SHOULDER")], arr[mp_index("RIGHT_ELBOW")], arr[mp_index("RIGHT_WRIST")]))

    # Shoulder angles
    angles.append(angle_between(arr[mp_index("LEFT_HIP")], arr[mp_index("LEFT_SHOULDER")], arr[mp_index("LEFT_ELBOW")]))
    angles.append(angle_between(arr[mp_index("RIGHT_HIP")], arr[mp_index("RIGHT_SHOULDER")], arr[mp_index("RIGHT_ELBOW")]))

    # Knee angles
    angles.append(angle_between(arr[mp_index("LEFT_HIP")], arr[mp_index("LEFT_KNEE")], arr[mp_index("LEFT_ANKLE")]))
    angles.append(angle_between(arr[mp_index("RIGHT_HIP")], arr[mp_index("RIGHT_KNEE")], arr[mp_index("RIGHT_ANKLE")]))

    # Hip angles
    angles.append(angle_between(arr[mp_index("LEFT_SHOULDER")], arr[mp_index("LEFT_HIP")], arr[mp_index("LEFT_KNEE")]))
    angles.append(angle_between(arr[mp_index("RIGHT_SHOULDER")], arr[mp_index("RIGHT_HIP")], arr[mp_index("RIGHT_KNEE")]))

    return np.array(angles)


# ---------------- Combined Feature Per Frame -------------------

def compute_features_for_frame(landmarks):
    """
    Combine:
    - 99 normalized coords (33 * 3)
    - 8 angle features
    Returns → 107-dim vector
    """
    if landmarks is None:
        return np.zeros(NUM_LM * FEATURES_PER_LM + 8, dtype=np.float32)

    coords = normalize_landmarks_simple(landmarks)
    angles = compute_angle_features(landmarks)

    return np.concatenate([coords, angles.astype(np.float32)])


# ---------------- Processing Entire JSON Files -------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def process_file(json_path, out_dir=PROC_DIR, seq_len=60):
    data = load_json(json_path)
    label = data.get("label", "unknown")
    frames = data["data"]

    feats = []
    for frame in frames:
        lm = frame.get("landmarks", None)
        feats.append(compute_features_for_frame(lm) if lm else None)

    # Fix missing frames by interpolation
    for i in range(len(feats)):
        if feats[i] is None:
            lo = i - 1
            while lo >= 0 and feats[lo] is None:
                lo -= 1
            hi = i + 1
            while hi < len(feats) and feats[hi] is None:
                hi += 1

            if lo >= 0 and hi < len(feats):
                feats[i] = (feats[lo] + feats[hi]) / 2
            elif lo >= 0:
                feats[i] = feats[lo]
            elif hi < len(feats):
                feats[i] = feats[hi]
            else:
                feats[i] = np.zeros(NUM_LM*FEATURES_PER_LM + 8)

    feats = np.array(feats)

    # sliding windows
    windows = []
    step = seq_len // 2

    for start in range(0, len(feats) - seq_len + 1, step):
        windows.append(feats[start:start+seq_len])

    if not windows:
        print(f"Skipping {json_path}: too few frames.")
        return

    windows = np.array(windows)

    # Save
    base = os.path.splitext(os.path.basename(json_path))[0]
    np.save(os.path.join(out_dir, f"{base}_X.npy"), windows)
    np.save(os.path.join(out_dir, f"{base}_y.npy"), np.array([label] * len(windows)))

    print(f"Processed {json_path} → {windows.shape}")


if __name__ == "__main__":
    files = glob(os.path.join(RAW_DIR, "*.json"))
    for f in tqdm(files):
        process_file(f)
