"""
guided_capture.py — Real-time pose-guided photo capture
-------------------------------------------------------
Uses the webcam + MediaPipe to guide the user into the correct pose:
  1. Full body visible (head to feet)
  2. Standing straight (vertical alignment)
  3. Arms slightly away from body (A-pose)
  4. Facing camera (front) or perpendicular (side)

Shows real-time feedback overlays and auto-captures when all checks pass.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        _POSE = mp.solutions.pose
        _DRAW = mp.solutions.drawing_utils
        _MODE = "legacy"
    else:
        _MODE = "tasks"
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    _MODE = None


@dataclass
class PoseCheck:
    full_body_visible: bool = False
    standing_straight: bool = False
    arms_away: bool = False
    facing_correct: bool = False
    centered: bool = False
    score: float = 0.0  # 0-1, how ready for capture

    @property
    def ready(self) -> bool:
        return all([self.full_body_visible, self.standing_straight,
                    self.arms_away, self.centered])

    @property
    def messages(self) -> list[tuple[str, bool]]:
        """Return (message, is_ok) pairs for UI display."""
        return [
            ("Full body visible (head to feet)", self.full_body_visible),
            ("Standing straight", self.standing_straight),
            ("Arms slightly away from body", self.arms_away),
            ("Body centered in frame", self.centered),
        ]


_LM_IDX = dict(
    nose=0, l_shoulder=11, r_shoulder=12, l_elbow=13, r_elbow=14,
    l_wrist=15, r_wrist=16, l_hip=23, r_hip=24, l_knee=25, r_knee=26,
    l_ankle=27, r_ankle=28, l_heel=29, r_heel=30
)


def check_pose(landmarks, img_h: int, img_w: int, view: str = "front") -> PoseCheck:
    """Analyze pose landmarks and return a PoseCheck with pass/fail for each criterion."""
    check = PoseCheck()

    if landmarks is None:
        return check

    # Extract key points as (x, y) in pixel coords
    pts = {}
    vis = {}
    for name, idx in _LM_IDX.items():
        lm = landmarks.landmark[idx]
        pts[name] = (lm.x * img_w, lm.y * img_h)
        vis[name] = lm.visibility

    # 1. Full body visible: head and ankles must be in frame with margin
    margin = 0.05
    head_ok = pts['nose'][1] > img_h * margin
    feet_ok = (pts['l_ankle'][1] < img_h * (1 - margin) and
               pts['r_ankle'][1] < img_h * (1 - margin))
    key_visible = all(vis[k] > 0.5 for k in
                      ['nose', 'l_shoulder', 'r_shoulder', 'l_hip', 'r_hip',
                       'l_ankle', 'r_ankle'])
    check.full_body_visible = head_ok and feet_ok and key_visible

    # 2. Standing straight: shoulders and hips should be roughly level,
    #    spine should be vertical
    sh_tilt = abs(pts['l_shoulder'][1] - pts['r_shoulder'][1]) / img_h
    hip_tilt = abs(pts['l_hip'][1] - pts['r_hip'][1]) / img_h
    # Spine vertical: midpoint of shoulders should be above midpoint of hips,
    # and horizontally aligned
    sh_mid_x = (pts['l_shoulder'][0] + pts['r_shoulder'][0]) / 2
    hip_mid_x = (pts['l_hip'][0] + pts['r_hip'][0]) / 2
    spine_lean = abs(sh_mid_x - hip_mid_x) / img_w
    check.standing_straight = (sh_tilt < 0.03 and hip_tilt < 0.03 and
                               spine_lean < 0.04)

    # 3. Arms away from body: elbows should be laterally outside the torso
    torso_left = min(pts['l_shoulder'][0], pts['l_hip'][0])
    torso_right = max(pts['r_shoulder'][0], pts['r_hip'][0])
    torso_w = torso_right - torso_left
    # Wrists should be at least 15% of torso width away from torso edge
    l_arm_out = pts['l_wrist'][0] < torso_left - torso_w * 0.1
    r_arm_out = pts['r_wrist'][0] > torso_right + torso_w * 0.1
    # Also check elbows aren't pressed against body
    l_elbow_out = pts['l_elbow'][0] < torso_left + torso_w * 0.1
    r_elbow_out = pts['r_elbow'][0] > torso_right - torso_w * 0.1
    check.arms_away = (l_arm_out or l_elbow_out) and (r_arm_out or r_elbow_out)

    # 4. Centered in frame
    body_cx = (sh_mid_x + hip_mid_x) / 2
    frame_cx = img_w / 2
    check.centered = abs(body_cx - frame_cx) < img_w * 0.15

    # Overall score
    scores = [check.full_body_visible, check.standing_straight,
              check.arms_away, check.centered]
    check.score = sum(scores) / len(scores)

    return check


def draw_guide_overlay(frame: np.ndarray, check: PoseCheck,
                       view: str = "front") -> np.ndarray:
    """Draw pose guidance overlay on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw body outline guide (where the person should stand)
    cx, cy = w // 2, h // 2
    body_w = int(w * 0.25)
    body_h = int(h * 0.85)
    top = cy - body_h // 2
    bot = cy + body_h // 2

    # Silhouette guide box
    guide_color = (0, 255, 0) if check.ready else (0, 200, 255)
    cv2.rectangle(overlay, (cx - body_w, top), (cx + body_w, bot),
                  guide_color, 2, cv2.LINE_AA)

    # Head circle
    head_r = int(body_w * 0.4)
    head_y = top + head_r + 10
    cv2.circle(overlay, (cx, head_y), head_r, guide_color, 2, cv2.LINE_AA)

    # Arm guides (A-pose)
    sh_y = top + int(body_h * 0.22)
    arm_end_y = top + int(body_h * 0.55)
    # Left arm
    cv2.line(overlay, (cx - body_w, sh_y),
             (cx - body_w - int(body_w * 0.6), arm_end_y),
             guide_color, 2, cv2.LINE_AA)
    # Right arm
    cv2.line(overlay, (cx + body_w, sh_y),
             (cx + body_w + int(body_w * 0.6), arm_end_y),
             guide_color, 2, cv2.LINE_AA)

    # Status panel at top
    panel_h = 30 + 25 * len(check.messages)
    cv2.rectangle(overlay, (10, 10), (w - 10, panel_h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (w - 10, panel_h), guide_color, 2)

    y_text = 35
    title = f"{'FRONT' if view == 'front' else 'SIDE'} VIEW — "
    title += "READY! Hold still..." if check.ready else "Adjust your position"
    cv2.putText(overlay, title, (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_text += 30

    for msg, ok in check.messages:
        color = (0, 255, 0) if ok else (0, 100, 255)
        icon = "[OK]" if ok else "[!!]"
        cv2.putText(overlay, f"{icon} {msg}", (25, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_text += 22

    # Blend overlay
    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return result


def extract_best_frame(video_path: str, view: str = "front") -> tuple:
    """Process a video and extract the best frame for the given view.

    For 'front': picks frame where shoulders are widest (facing camera)
    For 'side': picks frame where shoulders are narrowest (turned 90°)

    Returns (best_frame_bgr, PoseCheck) or (None, PoseCheck) if no good frame.
    """
    if not _MP_AVAILABLE or _MODE != "legacy":
        return None, PoseCheck()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, PoseCheck()

    best_frame = None
    best_score = -1.0
    best_check = PoseCheck()
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // 60)  # sample ~60 frames max

    with _POSE.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % sample_interval != 0:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            lm = results.pose_landmarks.landmark
            # Shoulder width in pixels (key signal for front vs side)
            l_sh = (lm[11].x * w, lm[11].y * h)
            r_sh = (lm[12].x * w, lm[12].y * h)
            sh_width = abs(l_sh[0] - r_sh[0])

            # Visibility of key landmarks
            key_vis = min(lm[11].visibility, lm[12].visibility,
                         lm[23].visibility, lm[24].visibility,
                         lm[27].visibility, lm[28].visibility)

            check = check_pose(results.pose_landmarks, h, w, view)

            if view == "front":
                # Front: maximize shoulder width (facing camera)
                # Also require good pose checks
                score = sh_width * (0.5 + 0.5 * check.score) * key_vis
            else:
                # Side: minimize shoulder width (turned 90°)
                # But still need full body visible
                if not check.full_body_visible:
                    continue
                score = (1.0 / max(sh_width, 1)) * 1000 * key_vis

            # Bonus for sharpness (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            score *= (1.0 + min(sharpness / 500.0, 1.0))  # mild sharpness bonus

            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_check = check

    cap.release()
    return best_frame, best_check


def validate_uploaded_image(bgr: np.ndarray, view: str = "front") -> PoseCheck:
    """Run pose checks on an already-uploaded image and return feedback."""
    if not _MP_AVAILABLE or bgr is None:
        return PoseCheck()

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if _MODE == "legacy":
        with _POSE.Pose(static_image_mode=True, model_complexity=2,
                        min_detection_confidence=0.4) as pose:
            results = pose.process(rgb)
            if results.pose_landmarks:
                return check_pose(results.pose_landmarks, h, w, view)
    return PoseCheck()
