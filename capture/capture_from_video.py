# capture/capture_from_video.py
import cv2, json, os
import mediapipe as mp
from datetime import datetime

mp_pose = mp.solutions.pose

def extract_from_video(video_path, label, out_dir="../data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    data = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                pts = [[l.x * w, l.y * h, l.z * w, l.visibility] for l in res.pose_landmarks.landmark]
            else:
                pts = None
            data.append({"landmarks": pts})
    cap.release()
    base = os.path.splitext(os.path.basename(video_path))[0]
    outname = f"{label}_{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(out_dir, outname), "w") as f:
        json.dump({"label": label, "data": data}, f)
    print("Wrote", outname)


