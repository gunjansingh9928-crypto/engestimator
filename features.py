"""Per-frame features from MediaPipe Face Mesh: head pose, EAR, blink rate."""
import cv2
import numpy as np
import mediapipe as mp

FEATURES = ["yaw", "pitch", "roll", "ear", "blink", "blink_rate", "face_found"]
GROUPS = {"head_pose": [0, 1, 2], "eye_closure": [3], "blink": [4, 5]}

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
POSE_IDX = [1, 152, 33, 263, 61, 291]  # nose, chin, eye corners, mouth corners
MODEL_3D = np.array([[0, 0, 0], [0, -63.6, -12.5], [-43.3, 32.7, -26],
                     [43.3, 32.7, -26], [-28.9, -28.9, -24.1],
                     [28.9, -28.9, -24.1]], dtype=np.float64)


def _ear(pts):
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return (a + b) / (2.0 * c + 1e-6)


def _head_pose(lm, w, h):
    img_pts = np.array([[lm[i].x * w, lm[i].y * h] for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img_pts, cam, np.zeros((4, 1)))
    if not ok:
        return 0.0, 0.0, 0.0
    angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]  # pitch, yaw, roll (deg)
    return angles[1], angles[0], angles[2]


def extract_clip_features(video_path, fps=5, seconds=10, ear_thresh=0.21, blink_window_s=2):
    """Return array (fps*seconds, len(FEATURES)). Frames without a face are zeros."""
    T = fps * seconds
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / fps)), 1)
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                           refine_landmarks=True)
    rows, i = [], 0
    while len(rows) < T:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                yaw, pitch, roll = _head_pose(lm, w, h)
                pts = lambda idx: np.array([[lm[j].x * w, lm[j].y * h] for j in idx])
                ear = (_ear(pts(LEFT_EYE)) + _ear(pts(RIGHT_EYE))) / 2
                rows.append([yaw, pitch, roll, ear, float(ear < ear_thresh), 0.0, 1.0])
            else:
                rows.append([0, 0, 0, 0, 0, 0, 0.0])
        i += 1
    cap.release(); mesh.close()

    arr = np.zeros((T, len(FEATURES)), dtype=np.float32)
    if rows:
        arr[:len(rows)] = np.array(rows, dtype=np.float32)
    # blink rate: blink onsets in a rolling window, scaled to blinks/minute
    closed = arr[:, 4]
    onsets = np.diff(closed, prepend=0) > 0
    win = max(int(blink_window_s * fps), 1)
    counts = np.convolve(onsets.astype(float), np.ones(win), mode="full")[:T]
    arr[:, 5] = counts * (60.0 / blink_window_s)
    return arr
