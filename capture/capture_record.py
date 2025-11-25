# capture/capture_record.py
import cv2, json, time, os
import mediapipe as mp
from datetime import datetime

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

OUT_DIR = "/Users/ankushaggarwal/Desktop/fitness-pose-project/data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

def landmarks_to_list(landmark, image_w, image_h):
    return [landmark.x * image_w, landmark.y * image_h, landmark.z * image_w, landmark.visibility]

def record(label: str, duration: int = 10):
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        start = time.time()
        data = []
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            image_h, image_w = frame.shape[:2]
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                pts = [landmarks_to_list(l, image_w, image_h) for l in results.pose_landmarks.landmark]
            else:
                pts = None
            timestamp = time.time()
            data.append({"timestamp": timestamp, "landmarks": pts})
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Label: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            cv2.imshow("Record (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    fname = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(OUT_DIR, fname), "w") as f:
        json.dump({"label": label, "data": data}, f)
    print("Saved", fname)

if __name__ == "__main__":
    label = input("Enter label (e.g., squat_correct, squat_knees_in): ").strip()
    duration = int(input("Duration seconds (default 10): ") or "10")
    record(label, duration)
