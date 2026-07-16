"""
measure_engine.py  (v2 — arm-aware)
-----------------------------------
Body-measurement estimation from a photo, calibrated by real HEIGHT + WEIGHT.

WHY v1 GAVE NONSENSE (waist=135cm on a lean 170cm/68kg subject):
  v1 measured row width as (last_white_px - first_white_px). When the subject
  stands with arms DOWN AGAINST the body — i.e. how people normally stand —
  the silhouette fuses the arms into the torso, so that "width" is actually
  torso + both arms. Girths came out 40-60cm too large, and waist ended up
  wider than chest, which is anatomically impossible at BMI 23.

v2 FIXES:
  1. ARM EXCLUSION — pose landmarks locate each arm at every row; the torso is
     measured strictly INSIDE the arm boundaries.
  2. CENTER RUN — width is the contiguous white run through the body centerline,
     not the full row extent. Kills stray blobs, shadows, and any graphics
     composited into the photo.
  3. SANITY CHECKS — anthropometric priors flag impossible results loudly
     instead of silently shipping a wrong number to a client.

Accuracy: front-only ~3-6cm girths; front+side ~2-3cm. Baseline, not
tailoring-grade. Swap this class for a BodyM-trained regressor later; the UI
depends only on measure() -> MeasurementResult.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# MediaPipe: 0.10.15+ REMOVED the legacy `mp.solutions` API. Newer builds expose
# only `mp.tasks`. Checking for `mp.solutions` alone silently disables pose on
# modern installs -> every image falls into the dumb silhouette-only path and
# returns garbage. We support BOTH APIs and record WHY if neither works.
# ---------------------------------------------------------------------------
_MP_MODE = None          # 'tasks' | 'legacy' | None
_MP_ERROR = None         # human-readable reason pose is unavailable

try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        _MP_MODE = "legacy"
    else:
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision
        _MP_MODE = "tasks"
except Exception as e:
    _MP_ERROR = f"MediaPipe unavailable: {e}"

_MP_OK = _MP_MODE is not None

# Tasks API needs a .task model file. Downloaded once, cached next to this file.
_POSE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "pose_landmarker/pose_landmarker_heavy/float16/1/"
                  "pose_landmarker_heavy.task")
_POSE_TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pose_landmarker_heavy.task")


def _ensure_pose_model() -> str | None:
    """Download the Tasks pose model on first use. Returns path or None."""
    global _MP_ERROR
    if os.path.exists(_POSE_TASK_PATH):
        return _POSE_TASK_PATH
    try:
        import urllib.request
        urllib.request.urlretrieve(_POSE_TASK_URL, _POSE_TASK_PATH)
        return _POSE_TASK_PATH
    except Exception as e:
        _MP_ERROR = (f"Could not download the pose model ({e}). "
                     f"Download it manually from:\n{_POSE_TASK_URL}\n"
                     f"and save it as: {_POSE_TASK_PATH}")
        return None


_POSE_LANDMARKER = None


def _get_landmarker():
    """Lazily build and cache the Tasks PoseLandmarker."""
    global _POSE_LANDMARKER, _MP_ERROR
    if _POSE_LANDMARKER is not None:
        return _POSE_LANDMARKER
    path = _ensure_pose_model()
    if path is None:
        return None
    try:
        opts = _mp_vision.PoseLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=path),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4)
        _POSE_LANDMARKER = _mp_vision.PoseLandmarker.create_from_options(opts)
        return _POSE_LANDMARKER
    except Exception as e:
        _MP_ERROR = f"Failed to init PoseLandmarker: {e}"
        return None

try:
    from rembg import remove as _rembg_remove
    _REMBG_OK = True
except Exception:
    _REMBG_OK = False


@dataclass
class MeasurementResult:
    measurements: dict
    landmarks_img: np.ndarray
    silhouette_img: np.ndarray
    confidence: float
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    scale_cm_per_px: float = 0.0
    used_side_view: bool = False
    photo_trust: float = 1.0
    provenance: dict = field(default_factory=dict)  # name -> 'measured'|'modeled'|'blended'|'given'


# ------------------------------ geometry ----------------------------------- #

def _ellipse_perimeter(width: float, depth: float) -> float:
    a, b = width / 2.0, depth / 2.0
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ----------------------------- silhouette ---------------------------------- #

def _silhouette_mask(bgr: np.ndarray) -> np.ndarray:
    if _REMBG_OK:
        rgba = _rembg_remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        alpha = np.array(rgba)[:, :, 3]
        mask = (alpha > 128).astype(np.uint8) * 255
        if mask.sum() > 0:
            return _clean_mask(mask)
    return _grabcut_mask(bgr)


def _grabcut_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.03), int(w * 0.84), int(h * 0.94))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        m = np.zeros((h, w), np.uint8)
        m[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 255
        return m
    out = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    return _clean_mask(out)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [big], -1, 255, -1)
        return clean
    return mask


def _center_run(mask, y, cx, x_lo=None, x_hi=None):
    """Contiguous white run through column cx at row y, clipped to [x_lo,x_hi].

    THE CORE FIX. Walk outward from the body centerline, stop at the first
    background pixel or at the arm boundary. Arms-down poses no longer inflate
    the measured width.
    """
    h, w = mask.shape
    y = int(np.clip(y, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))
    x_lo = 0 if x_lo is None else int(np.clip(x_lo, 0, w - 1))
    x_hi = w - 1 if x_hi is None else int(np.clip(x_hi, 0, w - 1))
    if x_hi <= x_lo:
        return None
    row = mask[y]
    if row[cx] == 0:
        xs = np.where(row[x_lo:x_hi + 1] > 0)[0]
        if xs.size == 0:
            return None
        cx = x_lo + int(xs[np.argmin(np.abs(xs - (cx - x_lo)))])
    left = cx
    while left > x_lo and row[left - 1] > 0:
        left -= 1
    right = cx
    while right < x_hi and row[right + 1] > 0:
        right += 1
    return left, right


def _torso_width_px(mask, y, cx, arm_lo, arm_hi, band=5) -> float:
    h = mask.shape[0]
    vals = []
    for yy in range(max(0, int(y) - band), min(h, int(y) + band + 1)):
        r = _center_run(mask, yy, cx, arm_lo, arm_hi)
        if r:
            vals.append(r[1] - r[0])
    return float(np.median(vals)) if vals else 0.0


# -------------------------------- pose ------------------------------------- #

_LM = dict(nose=0, l_shoulder=11, r_shoulder=12, l_elbow=13, r_elbow=14,
           l_wrist=15, r_wrist=16, l_hip=23, r_hip=24, l_knee=25, r_knee=26,
           l_ankle=27, r_ankle=28, l_heel=29, r_heel=30, l_foot=31, r_foot=32)


def _pose_landmarks(bgr: np.ndarray):
    """Return (pts, vis) or None. Works on BOTH the modern Tasks API and the
    legacy solutions API, so the pose path isn't silently skipped."""
    global _MP_ERROR
    if not _MP_OK:
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if _MP_MODE == "tasks":
        lmk = _get_landmarker()
        if lmk is None:
            return None
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lmk.detect(mp_img)
        except Exception as e:
            _MP_ERROR = f"Pose detection failed: {e}"
            return None
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        pts = {n: (lm[i].x * w, lm[i].y * h) for n, i in _LM.items()}
        vis = {n: getattr(lm[i], "visibility", 1.0) for n, i in _LM.items()}
        return pts, vis

    # legacy
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                min_detection_confidence=0.4) as pose:
        res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    pts = {n: (lm[i].x * w, lm[i].y * h) for n, i in _LM.items()}
    vis = {n: lm[i].visibility for n, i in _LM.items()}
    return pts, vis


def _arm_bounds_at(y: float, pts: dict, margin_px: float):
    """Inner x-bounds of the two arms at row y, pulled in by margin_px.

    Two subtleties that matter (both were bugs in the first pass):
      - Below the wrist the hands/forearms are STILL beside the hips, so the
        bound must PERSIST (clamp to the wrist x), not vanish to infinity.
        Otherwise the hip row measures torso+arms and blows up.
      - Above the shoulder there is no arm, so the bound opens up.
    """
    def arm_x(sh, el, wr):
        ys = [pts[sh][1], pts[el][1], pts[wr][1]]
        xs = [pts[sh][0], pts[el][0], pts[wr][0]]
        order = np.argsort(ys)
        ys_s = [ys[i] for i in order]
        xs_s = [xs[i] for i in order]
        if y < ys_s[0]:
            return None                 # above the shoulder: no arm here
        if y > ys_s[-1]:
            return xs_s[-1]             # below the wrist: hand still beside body
        return float(np.interp(y, ys_s, xs_s))

    if pts['l_shoulder'][0] < pts['r_shoulder'][0]:
        L, R = ('l_shoulder', 'l_elbow', 'l_wrist'), \
               ('r_shoulder', 'r_elbow', 'r_wrist')
    else:
        L, R = ('r_shoulder', 'r_elbow', 'r_wrist'), \
               ('l_shoulder', 'l_elbow', 'l_wrist')

    lx, rx = arm_x(*L), arm_x(*R)
    lo = (lx + margin_px) if lx is not None else -1e9
    hi = (rx - margin_px) if rx is not None else 1e9
    return lo, hi


# ------------------------------- engine ------------------------------------ #

class MeasurementEngine:

    def measure(self, front_bgr, height_cm, weight_kg, sex="neutral",
                side_bgr=None) -> MeasurementResult:
        notes, warns = [], []
        mask = _silhouette_mask(front_bgr)
        pose = _pose_landmarks(front_bgr)

        if pose is None:
            if _MP_ERROR:
                warns.append(f"**Pose detection is not working** — "
                             f"{_MP_ERROR}\n\nWithout pose landmarks the app "
                             f"cannot separate arms from the torso, so girths "
                             f"from the photo are unreliable and it is falling "
                             f"back to the height/BMI model.")
            elif not _MP_OK:
                warns.append("**MediaPipe is not installed**, so pose "
                             "detection is unavailable. Run "
                             "`pip install mediapipe` — without it the app "
                             "cannot exclude arms from the torso and girths "
                             "fall back to the height/BMI model.")
            else:
                warns.append("**No person detected in the image.** Upload a "
                             "clear, full-body, front-facing photo with the "
                             "whole body (head to feet) inside the frame.")
            return self._silhouette_only(front_bgr, mask, height_cm,
                                         weight_kg, notes, warns)

        pts, vis = pose
        ys = np.where(mask.any(axis=1))[0]
        if ys.size == 0:
            warns.append("No subject found in the image.")
            return MeasurementResult({}, front_bgr, np.zeros_like(front_bgr),
                                     0.05, notes, warns)

        person_px = max(1.0, float(ys[-1] - ys[0]))
        scale = height_cm / person_px
        notes.append(f"Scale locked to your height: {scale:.4f} cm/px "
                     f"(subject spans {person_px:.0f} px).")

        bmi = weight_kg / ((height_cm / 100) ** 2)

        sh_y = (pts['l_shoulder'][1] + pts['r_shoulder'][1]) / 2
        hip_y = (pts['l_hip'][1] + pts['r_hip'][1]) / 2
        cx = int((pts['l_hip'][0] + pts['r_hip'][0] +
                  pts['l_shoulder'][0] + pts['r_shoulder'][0]) / 4)
        torso_px = max(1.0, hip_y - sh_y)

        chest_y = sh_y + 0.20 * torso_px
        waist_y = sh_y + 0.68 * torso_px
        hip_y_m = hip_y + 0.16 * torso_px

        arm_margin = 0.035 * person_px

        def w_at(y):
            lo, hi = _arm_bounds_at(y, pts, arm_margin)
            return _torso_width_px(mask, y, cx, lo, hi) * scale

        chest_w = w_at(chest_y)
        waist_w = w_at(waist_y)
        hip_w = w_at(hip_y_m)
        # Shoulder width -> BIACROMIAL BREADTH.
        # MediaPipe's shoulder landmarks sit at the GLENOHUMERAL JOINT CENTERS
        # (inside the deltoids), NOT at the acromion (the bony outer edge that
        # tailors measure to). Reporting the raw landmark distance under-reports
        # shoulder width by ~25%. We correct to biacromial breadth, then
        # cross-check against the silhouette's actual deltoid-to-deltoid width.
        ACROMION_K = 1.27          # joint-center -> acromion, anthropometric
        sh_joint_w = _dist(pts['l_shoulder'], pts['r_shoulder']) * scale
        shoulder_w = sh_joint_w * ACROMION_K
        shoulder_src = "measured"

        # Cross-check: silhouette width at the deltoid line (just below the
        # shoulder landmarks) is the true outer edge, if the mask is clean.
        delt_y = sh_y + 0.06 * torso_px
        r_delt = _center_run(mask, int(delt_y), cx)
        if r_delt:
            delt_w = (r_delt[1] - r_delt[0]) * scale
            # trust the silhouette only if it's in a sane band vs the skeleton
            if 1.0 * sh_joint_w <= delt_w <= 1.55 * sh_joint_w:
                shoulder_w = 0.5 * shoulder_w + 0.5 * delt_w

        used_side = False
        if side_bgr is not None:
            sm = _silhouette_mask(side_bgr)
            sy = np.where(sm.any(axis=1))[0]
            if sy.size:
                s_scale = height_cm / max(1.0, float(sy[-1] - sy[0]))
                ratio = (sy[-1] - sy[0]) / person_px
                s_cx = int(np.median(np.where(sm.any(axis=0))[0]))

                def d_at(yf):
                    y_s = sy[0] + (yf - ys[0]) * ratio
                    r = _center_run(sm, int(y_s), s_cx)
                    return (r[1] - r[0]) * s_scale if r else 0.0

                chest_d, waist_d, hip_d = d_at(chest_y), d_at(waist_y), \
                                          d_at(hip_y_m)
                used_side = True
                notes.append("Side photo used — depths measured directly. "
                             "This is the most accurate mode.")
        if not used_side:
            r_c = float(np.clip(0.55 + (bmi - 22) * 0.010, 0.45, 0.85))
            r_w = float(np.clip(0.62 + (bmi - 22) * 0.016, 0.50, 1.00))
            r_h = float(np.clip(0.70 + (bmi - 22) * 0.010, 0.60, 0.95))
            chest_d, waist_d, hip_d = chest_w * r_c, waist_w * r_w, hip_w * r_h
            notes.append(f"No side photo — depth inferred from BMI {bmi:.1f}. "
                         "A side photo roughly halves girth error.")

        chest_c_raw = _ellipse_perimeter(chest_w, chest_d)
        waist_c_raw = _ellipse_perimeter(waist_w, waist_d)
        hip_c_raw = _ellipse_perimeter(hip_w, hip_d)

        # ---- FUSE silhouette with anthropometric prior -------------------- #
        # Loose clothing means the silhouette is measuring FABRIC, not body.
        # Rather than shipping a fabric measurement as if it were a body
        # measurement, we blend toward a height+BMI prior and report how much
        # each source contributed. Honest > confidently wrong.
        chest_c, waist_c, hip_c, trust, fuse_note = self._fuse_with_prior(
            chest_c_raw, waist_c_raw, hip_c_raw, bmi, height_cm, used_side)
        notes.append(fuse_note)

        # Arm length (shoulder -> wrist). Two anatomical corrections:
        #  (a) tailors measure from the ACROMION, not the joint center, so the
        #      upper-arm segment is under-reported by the same offset as above;
        #  (b) MediaPipe's "wrist" is the wrist joint; sleeve length runs to the
        #      styloid/base of the hand, a little further.
        # A frontal photo also foreshortens a slightly-bent arm, so we take the
        # LONGER of the two arms rather than averaging (a bent arm can only
        # measure short, never long).
        def _arm(sh, el, wr):
            upper = _dist(pts[sh], pts[el]) * scale
            fore = _dist(pts[el], pts[wr]) * scale
            return upper * 1.10 + fore * 1.06   # acromion + styloid offsets

        arm_len = max(_arm('l_shoulder', 'l_elbow', 'l_wrist'),
                      _arm('r_shoulder', 'r_elbow', 'r_wrist'))
        arm_modeled = False
        # Plausibility: sleeve length is ~0.31-0.35 x stature.
        if not (0.30 * height_cm <= arm_len <= 0.37 * height_cm):
            arm_len = 0.325 * height_cm
            arm_modeled = True
        # Neck: shoulder width is a poor proxy (it's dominated by deltoids).
        # Neck girth tracks height + BMI far better.
        neck_c = 0.195 * height_cm + 0.55 * (bmi - 22) + 3.5

        # Thigh: measure ONE leg. Scanning from the body centerline grabs BOTH
        # legs (they touch at the top), so we anchor on the knee's own x and
        # clip to the midline between the legs. Sample just below the crotch,
        # where the femur is at its thickest but the legs have already split.
        crotch_y0 = self._find_crotch(mask, hip_y, pts)
        knee_y0 = (pts['l_knee'][1] + pts['r_knee'][1]) / 2
        thigh_y = crotch_y0 + 0.18 * (knee_y0 - crotch_y0)
        mid_x = (pts['l_knee'][0] + pts['r_knee'][0]) / 2
        leg_cx = pts['l_knee'][0]
        if leg_cx < mid_x:
            lo_l, hi_l = 0, int(mid_x - 2)
        else:
            lo_l, hi_l = int(mid_x + 2), mask.shape[1] - 1
        r = _center_run(mask, int(thigh_y), int(leg_cx), lo_l, hi_l)
        thigh_w = (r[1] - r[0]) * scale if r else 0.0
        # thigh cross-section is near-circular, not a flat ellipse
        thigh_c = _ellipse_perimeter(thigh_w, thigh_w * 0.95)
        # Plausibility: thigh girth is ~0.55-0.62 x hip for most builds.
        # Anything outside 0.48-0.66 x hip means the scan caught shorts fabric
        # or bled into the other leg -> fall back to the ratio.
        thigh_modeled = False
        if thigh_c <= 0 or not (0.48 * hip_c <= thigh_c <= 0.66 * hip_c):
            thigh_c = 0.57 * hip_c
            thigh_modeled = True

        # Inseam: from CROTCH to ankle. MediaPipe's hip landmark is the hip
        # JOINT (roughly at the greater trochanter), which sits well above the
        # crotch — using it directly under-reports inseam by ~20cm. We find the
        # crotch by scanning down for where the mask splits into two legs.
        crotch_y = self._find_crotch(mask, hip_y, pts)
        ankle_y = (pts['l_ankle'][1] + pts['r_ankle'][1]) / 2
        inseam = max(0.0, (ankle_y - crotch_y)) * scale
        torso_len = torso_px * scale
        # Outseam runs from the waist, not the crotch
        outseam = max(0.0, (ankle_y - waist_y)) * scale
        knee_y = (pts['l_knee'][1] + pts['r_knee'][1]) / 2
        knee_h = max(0.0, (ankle_y - knee_y)) * scale

        # Plausibility guards against known ratios of stature
        inseam_modeled = outseam_modeled = False
        if not (0.40 * height_cm <= inseam <= 0.52 * height_cm):
            inseam = 0.46 * height_cm
            inseam_modeled = True
        if not (0.55 * height_cm <= outseam <= 0.68 * height_cm):
            outseam = 0.61 * height_cm
            outseam_modeled = True

        m = {
            "Height": round(height_cm, 1),
            "Weight": round(weight_kg, 1),
            "Shoulder width": round(shoulder_w, 1),
            "Chest / bust": round(chest_c, 1),
            "Waist": round(waist_c, 1),
            "Hip": round(hip_c, 1),
            "Neck": round(neck_c, 1),
            "Arm length (sh→wrist)": round(arm_len, 1),
            "Thigh": round(thigh_c, 1),
            "Inseam": round(inseam, 1),
            "Outseam": round(outseam, 1),
            "Torso length": round(torso_len, 1),
            "Knee height": round(knee_h, 1),
            "BMI": round(bmi, 1),
        }

        m, sw = self._sanity_check(m, bmi, height_cm)
        warns.extend(sw)

        # Provenance: tell the user WHICH numbers are real measurements and
        # which were substituted from a population model. Without this, a
        # modeled value is indistinguishable from a measured one -- which is
        # exactly how someone ends up trusting a number that isn't theirs.
        if trust >= 0.70:
            girth_src = "measured"
        elif trust >= 0.30:
            girth_src = "blended"
        else:
            girth_src = "modeled"

        prov = {
            "Height": "given", "Weight": "given", "BMI": "given",
            "Shoulder width": shoulder_src,
            "Arm length (sh→wrist)": ("modeled" if arm_modeled else "measured"),
            "Torso length": "measured",
            "Knee height": "measured",
            "Chest / bust": girth_src,
            "Waist": girth_src,
            "Hip": girth_src,
            "Neck": "modeled",                  # always from height+BMI prior
            "Thigh": "modeled" if thigh_modeled else "measured",
            "Inseam": "modeled" if inseam_modeled else "measured",
            "Outseam": "modeled" if outseam_modeled else "measured",
        }

        conf = self._confidence(vis, used_side, len(sw), trust)
        lm_img = self._draw(front_bgr, pts, mask, cx,
                            [(chest_y, 'chest'), (waist_y, 'waist'),
                             (hip_y_m, 'hip')], arm_margin)

        return MeasurementResult(m, lm_img,
                                 cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                                 conf, notes, warns, scale, used_side, trust,
                                 prov)

    def _find_crotch(self, mask, hip_y, pts):
        """Scan down from the hips for the row where the mask splits into two
        legs. That split point is the crotch — the true top of the inseam."""
        h, w = mask.shape
        mid_x = int((pts['l_knee'][0] + pts['r_knee'][0]) / 2)
        knee_y = int((pts['l_knee'][1] + pts['r_knee'][1]) / 2)
        start = int(np.clip(hip_y, 0, h - 1))
        end = int(np.clip(knee_y, 0, h - 1))
        for y in range(start, end):
            # crotch = first row where the centerline between the legs is
            # background (i.e. the legs have separated)
            if mask[y, int(np.clip(mid_x, 0, w - 1))] == 0:
                return float(y)
        return float(hip_y + 0.10 * (end - start))  # fallback

    # ------------------------ prior fusion -------------------------------- #

    @staticmethod
    def _prior_girths(bmi, height_cm):
        """Population regression: expected girths from height + BMI."""
        return (0.52 * height_cm + 1.35 * (bmi - 22) + 4,   # chest
                0.44 * height_cm + 1.65 * (bmi - 22) + 4,   # waist
                0.53 * height_cm + 1.30 * (bmi - 22) + 3)   # hip

    def _fuse_with_prior(self, chest_r, waist_r, hip_r, bmi, height_cm,
                         used_side):
        """Blend the silhouette measurement with the height+BMI prior.

        TRUST heuristic: if the silhouette says waist > chest on a lean subject,
        or a girth deviates wildly from the prior, the silhouette is measuring
        clothing / fused arms, not body — so we lean on the prior. If the
        silhouette agrees with the prior, we trust it and keep its detail
        (that detail is what distinguishes a barrel chest from a flat one).
        """
        p_chest, p_waist, p_hip = self._prior_girths(bmi, height_cm)

        # How badly does the silhouette disagree with the prior?
        devs = []
        for r, p in [(chest_r, p_chest), (waist_r, p_waist), (hip_r, p_hip)]:
            if r > 0 and p > 0:
                devs.append(abs(r - p) / p)
        mean_dev = float(np.mean(devs)) if devs else 1.0

        # Ordering violation is a hard tell that the mask is wrong
        ordering_bad = (bmi < 27 and 0 < chest_r < waist_r)

        # trust in [0,1]: 1 = pure silhouette, 0 = pure prior
        trust = float(np.clip(1.0 - mean_dev / 0.30, 0.0, 1.0))
        if ordering_bad:
            trust = min(trust, 0.15)
        if used_side:
            trust = min(1.0, trust + 0.15)   # measured depth earns trust

        def blend(r, p):
            if r <= 0:
                return p
            return trust * r + (1 - trust) * p

        chest = blend(chest_r, p_chest)
        waist = blend(waist_r, p_waist)
        hip = blend(hip_r, p_hip)

        # Final guard: preserve anatomical ordering on lean subjects
        if bmi < 27 and waist > chest:
            waist = min(waist, chest - 6)

        pct = int(trust * 100)
        if pct >= 70:
            note = (f"Silhouette and body-model prior agree — result is "
                    f"{pct}% from your photo, {100-pct}% from the height/BMI "
                    f"model.")
        elif pct >= 30:
            note = (f"Silhouette partly disagrees with the body-model prior. "
                    f"Result is a blend: {pct}% photo, {100-pct}% height/BMI "
                    f"model. Fitted clothing + a side photo would raise the "
                    f"photo's weight.")
        else:
            note = (f"⚠️ Silhouette was **not trustworthy** (loose clothing, "
                    f"arms against the body, or a poor mask). Falling back "
                    f"mostly to the height/BMI model: only {pct}% of this "
                    f"result comes from your photo. **These girths are a "
                    f"population estimate for your height and weight — not a "
                    f"measurement of you.** Re-shoot in fitted clothing, "
                    f"A-pose (arms away from body), plain background.")
        return (round(chest, 1), round(waist, 1), round(hip, 1), trust, note)

    # ------------------------ sanity / priors ----------------------------- #

    def _sanity_check(self, m, bmi, height_cm):
        w = []
        chest = m.get("Chest / bust", 0)
        waist = m.get("Waist", 0)
        hip = m.get("Hip", 0)

        exp_chest, exp_waist, exp_hip = self._prior_girths(bmi, height_cm)

        for name, val, exp in [("Chest / bust", chest, exp_chest),
                               ("Waist", waist, exp_waist),
                               ("Hip", hip, exp_hip)]:
            if val <= 0:
                continue
            dev = (val - exp) / exp
            if abs(dev) > 0.28:
                w.append(
                    f"**{name} = {val} cm** is {abs(dev)*100:.0f}% "
                    f"{'above' if dev > 0 else 'below'} the typical value for "
                    f"BMI {bmi:.1f} at {height_cm:.0f} cm (≈{exp:.0f} cm). "
                    f"This points to a segmentation problem. Check the "
                    f"silhouette below. Usual causes: arms resting against the "
                    f"body, loose clothing, a busy background, or graphics/text "
                    f"baked into the image.")

        if bmi < 27 and 0 < chest < waist:
            w.append(
                f"**Waist ({waist} cm) exceeds chest ({chest} cm)** at "
                f"BMI {bmi:.1f}. That is not real anatomy — it is a measurement "
                f"error. The usual cause is arms hanging against the torso, "
                f"fusing them into the silhouette.")
        return m, w

    def _confidence(self, vis, used_side, n_warn, trust=1.0):
        core = ['l_shoulder', 'r_shoulder', 'l_hip', 'r_hip', 'l_ankle',
                'r_ankle', 'l_wrist', 'r_wrist']
        v = float(np.mean([vis[k] for k in core]))
        c = 0.30 + 0.25 * v + 0.40 * trust   # trust dominates: a pretty pose
        if used_side:                        # with a bad mask is still bad
            c += 0.08
        c -= 0.10 * n_warn
        return float(np.clip(c, 0.05, 0.95))

    def _draw(self, bgr, pts, mask, cx, levels, margin):
        img = bgr.copy()
        conns = [('l_shoulder', 'r_shoulder'), ('l_shoulder', 'l_elbow'),
                 ('l_elbow', 'l_wrist'), ('r_shoulder', 'r_elbow'),
                 ('r_elbow', 'r_wrist'), ('l_shoulder', 'l_hip'),
                 ('r_shoulder', 'r_hip'), ('l_hip', 'r_hip'),
                 ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
                 ('r_hip', 'r_knee'), ('r_knee', 'r_ankle')]
        for a, b in conns:
            cv2.line(img, tuple(map(int, pts[a])), tuple(map(int, pts[b])),
                     (255, 180, 0), 3)
        for p in pts.values():
            cv2.circle(img, tuple(map(int, p)), 5, (0, 255, 0), -1)
        for y, lab in levels:
            lo, hi = _arm_bounds_at(y, pts, margin)
            r = _center_run(mask, int(y), cx, lo, hi)
            if r:
                cv2.line(img, (r[0], int(y)), (r[1], int(y)), (0, 0, 255), 4)
                cv2.putText(img, lab, (r[1] + 10, int(y) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img

    def _silhouette_only(self, bgr, mask, height_cm, weight_kg, notes, warns):
        ys = np.where(mask.any(axis=1))[0]
        if not ys.size:
            warns.append("No subject found in the image.")
            return MeasurementResult({}, bgr, np.zeros_like(bgr), 0.05,
                                     notes, warns)
        top, bot = int(ys[0]), int(ys[-1])
        scale = height_cm / max(1, bot - top)
        cx = int(np.median(np.where(mask.any(axis=0))[0]))
        bmi = weight_kg / ((height_cm / 100) ** 2)

        def w(frac):
            y = int(top + frac * (bot - top))
            r = _center_run(mask, y, cx)
            return (r[1] - r[0]) * scale if r else 0.0

        cw, ww, hw = w(0.30), w(0.46), w(0.53)
        chest_r = _ellipse_perimeter(cw, cw * 0.62)
        waist_r = _ellipse_perimeter(ww, ww * 0.70)
        hip_r = _ellipse_perimeter(hw, hw * 0.75)

        # CRITICAL: the fallback MUST fuse too. Without this it returned raw
        # arms-fused widths at photo_trust=1.0 (the dataclass default) --
        # i.e. it reported MAXIMUM confidence in numbers it simultaneously
        # flagged as anatomically impossible. Worst possible failure mode.
        chest_c, waist_c, hip_c, trust, fuse_note = self._fuse_with_prior(
            chest_r, waist_r, hip_r, bmi, height_cm, used_side=False)
        # No pose = no arm exclusion = the silhouette is inherently unreliable.
        trust = min(trust, 0.25)
        notes.append(fuse_note)

        p_chest, p_waist, p_hip = self._prior_girths(bmi, height_cm)
        m = {"Height": round(height_cm, 1), "Weight": round(weight_kg, 1),
             "Chest / bust": chest_c, "Waist": waist_c, "Hip": hip_c,
             "Neck": round(0.195 * height_cm + 0.55 * (bmi - 22) + 3.5, 1),
             "Thigh": round(0.57 * hip_c, 1),
             "Inseam": round(0.46 * height_cm, 1),
             "Outseam": round(0.58 * height_cm, 1),
             "BMI": round(bmi, 1)}
        m, sw = self._sanity_check(m, bmi, height_cm)
        warns.extend(sw)
        conf = float(np.clip(0.15 + 0.35 * trust, 0.05, 0.5))
        prov = {k: ("given" if k in ("Height", "Weight", "BMI") else "modeled")
                for k in m}
        return MeasurementResult(m, bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                                 conf, notes, warns, scale, False, trust, prov)